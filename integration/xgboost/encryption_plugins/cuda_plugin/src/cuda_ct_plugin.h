/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDA_CT_PLUGIN_H
#define CUDA_CT_PLUGIN_H

#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <chrono>

#include "paillier.h"
#include "base_plugin.h"
#include "local_plugin.h"
#include "endec.h"

#define PRECISION 1e9
#define TIME

class Timer2 {
public:
    Timer2() : start_time_(), end_time_() {}

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    double duration() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

namespace nvflare {

class Context {
  private:
    std::map<int, std::vector<CgbnPair> >* _bin_table;
    std::vector<int>* _bin_length;
    int map_it_offset = 0;

  public:
    explicit Context(
      std::map<int, std::vector<CgbnPair> >& bin_table,
      std::vector<int>& bin_length
    ) {
      _bin_table = &bin_table;
      _bin_length = &bin_length;
    }

    std::vector<CgbnPair> get_next_tuple(int& bid, int tuple_length = 2) {
      std::vector<CgbnPair> result;
      const auto bin_table_size = _bin_table->size(); 

      for (auto i = map_it_offset; i < bin_table_size; ++i) {
          auto it = _bin_table->begin();
          std::advance(it, i);
          int key = it->first;

          if (_bin_length->at(i) >= tuple_length) {
            for (int j = 0; j < tuple_length; ++j) {
              result.push_back(it->second.at(_bin_length->at(i) - tuple_length + j));
            }
            //std::cout << "before bin len " << i << " was " << _bin_length->at(i) << std::endl;
            _bin_length->at(i) -= tuple_length;
            //std::cout << "after bin len " << i << " is " << _bin_length->at(i) << std::endl;
            bid = key;
            map_it_offset = i;
            break;
          }
      }
      return result;
    }
};

// Define a structured header for the buffer
struct BufferHeader2 {
  bool has_key;
  size_t key_size;
  size_t rand_seed_size;
};

class CUDACTPlugin: public LocalPlugin {
  private:
    PaillierCipher<bits>* paillier_cipher_ptr_ = nullptr;
    CgbnPair* encrypted_gh_pairs_ = nullptr;
    Endec* endec_ptr_ = nullptr;
    int num_gh_pair_ = 0;

  public:
    explicit CUDACTPlugin(std::vector<std::pair<std::string_view, std::string_view> > const &args): LocalPlugin(args) {
      bool fix_seed = get_bool(args, "fix_seed");
      paillier_cipher_ptr_ = new PaillierCipher<bits>(bits/2, fix_seed, debug_);
      //paillier_cipher_ptr_->genKeypair(); // TODO remove this debug
      encrypted_gh_pairs_ = nullptr;
      
    }

    ~CUDACTPlugin() {
      delete paillier_cipher_ptr_;
      if (endec_ptr_ != nullptr) {
        delete endec_ptr_;
        endec_ptr_ = nullptr;
      }
    }

    void setGHPairs() {
      if (debug_) std::cout << "setGHPairs is called" << std::endl;
      const std::uint8_t* pointer = encrypted_gh_.data();

      // Retrieve header
      BufferHeader2 header;
      std::memcpy(&header, pointer, sizeof(BufferHeader2));
      pointer += sizeof(BufferHeader2);

      // Get key and n (if present)
      cgbn_mem_t<bits>* key_ptr;
      if (header.has_key) {
        mpz_t n;
        mpz_init(n);
        key_ptr = (cgbn_mem_t<bits>* )malloc(header.key_size);
        if (!key_ptr) {
          std::cout << "bad alloc with key_ptr" << std::endl;
          throw std::bad_alloc();
        }
        memcpy(key_ptr, pointer, header.key_size);
        store2Gmp(n, key_ptr);
        pointer += header.key_size;

        if (header.rand_seed_size != sizeof(uint64_t)) {
          free(key_ptr);
          mpz_clear(n);
          std::cout << "rand_seed_size " << header.rand_seed_size << " is wrong " << std::endl;
          throw std::runtime_error("Invalid random seed size");
        }
        uint64_t rand_seed;
        memcpy(&rand_seed, pointer, header.rand_seed_size);
        pointer += header.rand_seed_size;

        if (!paillier_cipher_ptr_->has_pub_key) {
          paillier_cipher_ptr_->set_pub_key(n, rand_seed);
        }
        mpz_clear(n);
        free(key_ptr);
      }

      // Access payload
      std::vector<std::uint8_t> payload(pointer, pointer + (encrypted_gh_.size() - (pointer - encrypted_gh_.data())));

      num_gh_pair_ = payload.size() / sizeof(CgbnPair);

      // TODO make either CPU or GPU
      //ck(cudaMalloc((void **)&encrypted_gh_pairs_, payload.size()));
      //cudaMemcpy(encrypted_gh_pairs_, payload.data(), payload.size(), cudaMemcpyHostToDevice);
      encrypted_gh_pairs_ = (CgbnPair*)malloc(payload.size());
      memcpy(encrypted_gh_pairs_, payload.data(), payload.size());
    }

    void clearGHPairs() {
      if (debug_) std::cout << "clearGHPairs is called" << std::endl;
      if (encrypted_gh_pairs_) {
        //cudaFree(encrypted_gh_pairs_);
        free(encrypted_gh_pairs_);
        encrypted_gh_pairs_ = nullptr;
      }
      if (debug_) std::cout << "clearGHPairs is finished" << std::endl;
    }

    Buffer createBuffer(
      bool has_key_flag,
      cgbn_mem_t<bits>* key_ptr,
      size_t key_size,
      uint64_t rand_seed,
      size_t rand_seed_size,
      cgbn_mem_t<bits>* d_ciphers_ptr,
      size_t payload_size
    ) {
        if (debug_) std::cout << "createBuffer is called" << std::endl;
        // Calculate header size and total buffer size
        size_t header_size = sizeof(BufferHeader2);
        size_t mem_size = header_size + key_size + rand_seed_size + payload_size;

        // Allocate buffer
        void* buffer = malloc(mem_size);
        if (!buffer) {
          std::cout << "bad alloc with buffer" << std::endl;
          throw std::bad_alloc();
        }

        // Construct header
        BufferHeader2 header;
        header.has_key = has_key_flag;
        header.key_size = key_size;
        header.rand_seed_size = rand_seed_size;

        // Copy header to buffer
        memcpy(buffer, &header, header_size);

        // Copy the key (if present)
        if (has_key_flag) {
          memcpy((char*)buffer + header_size, key_ptr, key_size);
          memcpy((char*)buffer + header_size + key_size, &rand_seed, rand_seed_size);
        }

        // Copy the payload
        cudaMemcpy((char*)buffer + header_size + key_size + rand_seed_size, d_ciphers_ptr, payload_size, cudaMemcpyDeviceToHost);

        Buffer result(buffer, mem_size, true);

        return result;
    }

    Buffer EncryptVector(const std::vector<double>& cleartext) override {
      if (debug_) std::cout << "Calling EncryptVector with count " << cleartext.size() << std::endl;
      if (endec_ptr_ != nullptr) {
        delete endec_ptr_;
      }
      endec_ptr_ = new Endec(PRECISION, debug_);

      size_t count = cleartext.size();
      int byte_length = bits / 8;
      size_t mem_size = sizeof(cgbn_mem_t<bits>) * count;
      cgbn_mem_t<bits>* h_ptr=(cgbn_mem_t<bits>* )malloc(mem_size);
      if (debug_) std::cout << "h_ptr size is " << mem_size << " indata size is " << count * byte_length << std::endl;
      for (size_t i = 0; i < count; ++i) {
        mpz_t n;
        mpz_init(n);
        //std::cout << "before encode " << i << " : " << cleartext[i] << std::endl;
        endec_ptr_->encode(n, cleartext[i]);
        store2Cgbn(h_ptr + i, n);

        mpz_clear(n);
      }

      cgbn_mem_t<bits>* d_plains_ptr;
      cgbn_mem_t<bits>* d_ciphers_ptr;
      ck(cudaMalloc((void **)&d_plains_ptr, mem_size));
      ck(cudaMalloc((void **)&d_ciphers_ptr, mem_size));
      cudaMemcpy(d_plains_ptr, h_ptr, mem_size, cudaMemcpyHostToDevice);

      if (!paillier_cipher_ptr_->has_prv_key) {
#ifdef TIME
      CudaTimer cuda_timer(0);
      float gen_time=0;
      cuda_timer.start();
#endif
        if (debug_) std::cout<<"Gen KeyPair with bits: " << bits << std::endl;
        paillier_cipher_ptr_->genKeypair();
#ifdef TIME
      gen_time += cuda_timer.stop();
      std::cout<<"Gen KeyPair Time "<< gen_time <<" MS"<<std::endl;
#endif
      }

      paillier_cipher_ptr_->encrypt<TPI,TPB>(d_plains_ptr, d_ciphers_ptr, count);

      // get pub_key n
      mpz_t n;
      mpz_init(n);
      size_t key_size = sizeof(cgbn_mem_t<bits>);
      paillier_cipher_ptr_->getN(n);
      store2Cgbn(h_ptr, n);
      mpz_clear(n);

      // get rand_seed
      size_t rand_seed_size = sizeof(uint64_t);
      uint64_t rand_seed = paillier_cipher_ptr_->get_rand_seed();

      Buffer result = createBuffer(true, h_ptr, key_size, rand_seed, rand_seed_size, d_ciphers_ptr, mem_size);

      cudaFree(d_plains_ptr);
      cudaFree(d_ciphers_ptr);
      free(h_ptr);

      return result;
    }

    std::vector<double> DecryptVector(const std::vector<Buffer>& ciphertext) override {
      if (debug_) std::cout << "Calling DecryptVector" << std::endl;
      size_t mem_size = 0;
      for (int i = 0; i < ciphertext.size(); ++i) {
        mem_size += ciphertext[i].buf_size;
        if (ciphertext[i].buf_size != 2 * sizeof(cgbn_mem_t<bits>)) {
          std::cout << "buf_size is " << ciphertext[i].buf_size << std::endl;
          std::cout << "expected buf_size is " << 2 * sizeof(cgbn_mem_t<bits>) << std::endl;
          std::cout << "Fatal Error" << std::endl;
        }
      }

      size_t count = mem_size / sizeof(cgbn_mem_t<bits>);
      cgbn_mem_t<bits>* h_ptr=(cgbn_mem_t<bits>* )malloc(mem_size);
      if (debug_) std::cout << "h_ptr size is " << mem_size << " how many gh is " << count << std::endl;
      

      cgbn_mem_t<bits>* d_plains_ptr;
      cgbn_mem_t<bits>* d_ciphers_ptr;
      ck(cudaMalloc((void **)&d_plains_ptr, mem_size));
      ck(cudaMalloc((void **)&d_ciphers_ptr, mem_size));
      
      size_t offset = 0;
      for (int i = 0; i < ciphertext.size(); ++i) {
        cudaMemcpy(d_ciphers_ptr + offset, ciphertext[i].buffer, ciphertext[i].buf_size, cudaMemcpyHostToDevice);
        offset += ciphertext[i].buf_size / sizeof(cgbn_mem_t<bits>);
      }

      if (!paillier_cipher_ptr_->has_prv_key) {
        std::cout << "Can't call DecryptVector if paillier does not have private key." << std::endl;
        throw std::runtime_error("Can't call DecryptVector if paillier does not have private key.");
      }


      paillier_cipher_ptr_->decrypt<TPI,TPB>(d_ciphers_ptr, d_plains_ptr, count);
      cudaMemcpy(h_ptr, d_plains_ptr, mem_size, cudaMemcpyDeviceToHost);

      std::vector<double> result;
      for (size_t i = 0; i < count; ++i) {
        mpz_t n;
        mpz_init(n);
        store2Gmp(n, h_ptr + i);
        double output_num = endec_ptr_->decode(n);
        result.push_back(output_num);
        std::cout << "decrypted result " << i << " : " << output_num << std::endl;
        mpz_clear(n);
      }
      cudaFree(d_plains_ptr);
      cudaFree(d_ciphers_ptr);
      free(h_ptr);
      return result;
    }

    void fillArray(
      CgbnPair* cell_table,
      std::map<int, std::vector<CgbnPair> >& bin_table,
      std::vector<int>& bin_length,
      std::vector<int>& rbt,
      int num_rows,
      int num_cols,
      int &last_row_used,
      int &last_col_used,
      int &num_tuples_filled,
      int tuple_length = 2
    ) {
      last_row_used = -1;
      last_col_used = -1;
      int num_tuples_per_row = num_cols / tuple_length;

      for (auto i = 0; i < num_rows; ++i) {
        for (auto j = 0; j < num_tuples_per_row; ++j) {
          rbt[i * num_tuples_per_row + j] = -1;
        }
      }

      double time = 0;
      Timer2 timer;
      Context ctx = Context(bin_table, bin_length);
      for (auto row = 0; row < num_rows; ++row) {
        for (auto t = 0; t < num_tuples_per_row; ++t) {
          int col = t * tuple_length;
          int bid = 0;
          //std::cout << "try to get a tuple for row " << row << " t " << t << std::endl;
          timer.start();
          std::vector<CgbnPair> next_tuple = ctx.get_next_tuple(bid, tuple_length);
          timer.stop();
          std::cout << "time to get a tuple is " << timer.duration() << " US " << std::endl;
          time += timer.duration();
          if (next_tuple.size() == 0) {
            std::cout << "time to get all tuple is " << time << " US " << std::endl;
            return;
          }
          timer.start();
          //std::cout << "gotten a tuple for row " << row << " t " << t << std::endl;
          for (auto i = 0; i < tuple_length; ++i) {
            cell_table[row * num_cols + col + i] = next_tuple[i];
          }
          num_tuples_filled += 1;

          last_row_used = row;
          last_col_used = col;
          rbt[row * num_tuples_per_row + t] = bid;
          timer.stop();
          std::cout << "time to set a tuple in table " << timer.duration() << " US " << std::endl;
        }
      }

    }

    void processResult(
      CgbnPair* cell_table,
      std::map<int, std::vector<CgbnPair> >& bin_table,
      std::vector<int>& bin_length,
      std::vector<int>& rbt,
      int num_rows,
      int num_cols,
      int tuple_length = 2
      ) {

      int num_tuples_per_row = num_cols / tuple_length;
      for (auto i = 0; i < num_rows; ++i) {
        for (auto j = 0; j < num_tuples_per_row; ++j) {
          int bid = rbt[i * num_tuples_per_row + j];
          if (bid < 0) {
            return;
          }
          bin_table[bid][bin_length[bid]] = cell_table[i * num_cols + j * tuple_length];
          bin_length[bid] += 1;
        }
      }

    }

    void printBinTable(std::map<int, std::vector<CgbnPair> > & bin_table, std::vector<int> bin_length) {
      if (!endec_ptr_) {
        endec_ptr_ = new Endec(PRECISION, debug_);
      }
      cgbn_mem_t<bits>* d_ciphers_ptr;
      cgbn_mem_t<bits>* d_plains_ptr;
      cgbn_mem_t<bits>* h_ptr = (cgbn_mem_t<bits> *)malloc(sizeof(CgbnPair));
      
      ck(cudaMalloc((void **)&d_ciphers_ptr, sizeof(CgbnPair)));
      

      ck(cudaMalloc((void **)&d_plains_ptr, sizeof(CgbnPair)));
      

      for (auto &pair: bin_table) {
        
        for (auto i = 0; i < bin_length[pair.first]; ++i) {
          cgbn_mem_t<bits>* ptr = &pair.second[i].g;
          printCgbn(ptr, 1);
          ptr = &pair.second[i].h;
          printCgbn(ptr, 1);
          ck(cudaMemcpy(d_ciphers_ptr, &pair.second[i], sizeof(CgbnPair), cudaMemcpyHostToDevice));
          printDevCgbn(d_ciphers_ptr, 2, "cipher");

          paillier_cipher_ptr_->decrypt<TPI,TPB>(d_ciphers_ptr, d_plains_ptr, 2);
          printDevCgbn(d_plains_ptr, 2, "plain");
          ck(cudaMemcpy(h_ptr, d_plains_ptr, sizeof(CgbnPair), cudaMemcpyDeviceToHost));
          
      
          for (size_t j = 0; j < 2; ++j) {
            mpz_t n;
            mpz_init(n);
            
            store2Gmp(n, h_ptr + j);
            
            double output_num = endec_ptr_->decode(n);
            
            std::cout << "bin_table[" << pair.first << "][" << i << "]" << "[" << j << "]:" << output_num << std::endl;
            mpz_clear(n);
          }
        }
      }
      free(h_ptr);
      cudaFree(d_ciphers_ptr);
      cudaFree(d_plains_ptr);
    }

    void printEncGHPairs() {
      if (!endec_ptr_) {
        endec_ptr_ = new Endec(PRECISION, debug_);
      }
      cgbn_mem_t<bits>* d_ciphers_ptr;
      cgbn_mem_t<bits>* d_plains_ptr;
      cgbn_mem_t<bits>* h_ptr = (cgbn_mem_t<bits> *)malloc(sizeof(CgbnPair));
      cudaMalloc((void **)&d_ciphers_ptr, sizeof(CgbnPair));
      cudaMalloc((void **)&d_plains_ptr, sizeof(CgbnPair));
      for (auto i = 0; i < num_gh_pair_; ++i) {
        cudaMemcpy(d_ciphers_ptr, &encrypted_gh_pairs_[i], sizeof(CgbnPair), cudaMemcpyHostToDevice);
        printDevCgbn(d_ciphers_ptr, 2, "cipher");
        paillier_cipher_ptr_->decrypt<TPI,TPB>(d_ciphers_ptr, d_plains_ptr, 2);
        printDevCgbn(d_plains_ptr, 2, "plain");
        cudaMemcpy(h_ptr, d_plains_ptr, sizeof(CgbnPair), cudaMemcpyDeviceToHost);

        for (size_t j = 0; j < 2; ++j) {
          mpz_t n;
          mpz_init(n);
          store2Gmp(n, h_ptr + j);
          double output_num = endec_ptr_->decode(n);
          std::cout << "enc_pair["  << i << "]" << "[" << j << "]:" << output_num << std::endl;
          mpz_clear(n);
        }
      }
      free(h_ptr);
      cudaFree(d_ciphers_ptr);
      cudaFree(d_plains_ptr);
    }

    std::map<int, Buffer> AddGHPairs(const std::map<int, std::vector<int> >& sample_ids) override{
      if (debug_) std::cout << "Calling AddGHPairs with sample_ids size " << sample_ids.size() << std::endl;
      if (!encrypted_gh_pairs_) {
        setGHPairs();
      }
      std::map<int, Buffer> result;
      size_t mem_size = sizeof(CgbnPair);

      if (!paillier_cipher_ptr_->has_pub_key) {
        std::cout << "Can't call AddGHPairs if paillier does not have public key." << std::endl;
        throw std::runtime_error("Can't call AddGHPairs if paillier does not have public key.");
      }

      int IPB = TPB / TPI;
      // TODO add memory limit
      int max_blocks = 4096; // limitation of hardware (GPU); max blocks is 2147483647, but there is memory limitation
      int max_num_of_instances_per_launch = IPB * max_blocks; // maximum numbers can be processed in a single launch

      std::cout << "Preparing bin_length and bin_table" << std::endl;

#ifdef TIME
      Timer2 timer;
      timer.start();
#endif
      std::vector<int> bin_length;
      std::map<int, std::vector<CgbnPair> > bin_table;
      int num_pairs = 0;
      for (auto& pair: sample_ids) {
        //std::cout << "working on " << pair.first << " with size " << pair.second.size() << std::endl;
        bin_length.push_back(pair.second.size());
        bin_table[pair.first] = std::vector<CgbnPair>();
        if (pair.second.size() == 0) {
          bin_table[pair.first].push_back(paillier_cipher_ptr_->get_encrypted_zero());
        } else {
          for (auto i = 0; i < pair.second.size(); ++i) {
            //std::cout << "working on " << pair.first << " with id " << i << std::endl;
            bin_table[pair.first].push_back(encrypted_gh_pairs_[pair.second[i]]);
            //std::cout << "end working on " << pair.first << " with id " << i << std::endl;
          }
          num_pairs += pair.second.size();
        }
        //std::cout << "end working on " << pair.first << std::endl;
      }

      //printCgbn((cgbn_mem_t<bits>*)encrypted_gh_pairs_, 2 * num_pairs);
      //printEncGHPairs();

      std::cout << "Finished preparing bin_length and bin_table, num_pairs is " << num_pairs << " num bins is " << sample_ids.size() << std::endl;
      //printBinTable(bin_table, bin_length);

#ifdef TIME
      timer.stop();
      std::cout<<"prepare bin_table/bin_length Time "<< timer.duration() <<" US"<<std::endl;
#endif

#ifdef TIME
      timer.start();
#endif

      int tuple_length = 2;
      int num_tuples_per_row = std::min(num_pairs, max_num_of_instances_per_launch) / tuple_length; 
      int num_cols = num_tuples_per_row * tuple_length; // needs to be a multiple of tuple
      int num_rows = (num_pairs - 1) / num_cols + 1;
      //int num_rows = 536870912; // 16 G / num_cols / 2
      std::cout << "max_num_of_instances_per_launch: " << max_num_of_instances_per_launch << " num_cols: " << num_cols << " num_rows: " << num_rows << std::endl;

      CgbnPair* cell_table;
      CgbnPair* d_cell_table;
      size_t table_size = sizeof(CgbnPair) * num_cols * num_rows;
      std::cout << "table mem size is " << table_size << std::endl;
      cell_table = (CgbnPair*)malloc(table_size);
      ck(cudaMalloc((void **)&d_cell_table, table_size));

      std::vector<int> rbt;
      for (auto i = 0; i < num_rows; ++i) {
        for (auto j = 0; j < num_tuples_per_row; ++j) {
          rbt.push_back(-1);
        }
      }

#ifdef TIME
      timer.stop();
      std::cout<<"malloc/cudaMalloc Time "<< timer.duration() <<" US"<<std::endl;
#endif

      int last_row = 0;
      int last_col = 0;
      int num_tuples_in_table = 0;
      int reduce_round = 0;

      while (true) {
        num_tuples_in_table = 0;

#ifdef TIME
        timer.start();
#endif
        std::cout << "Start fillArray for reduce_round " << reduce_round << std::endl;
        fillArray(cell_table, bin_table, bin_length, rbt, num_rows, num_cols, last_row, last_col, num_tuples_in_table, tuple_length);
        std::cout << "End fillArray for reduce_round " << reduce_round << std::endl;
        std::cout << "last row " << last_row << " last col " << last_col << " num_tuples_in_table " << num_tuples_in_table << std::endl;

#ifdef TIME
        timer.stop();
        std::cout<<"fillArray Time "<< timer.duration() <<" US"<<std::endl;
#endif

        if (last_row < 0) {
          break;
        }
        cudaMemcpy(d_cell_table, cell_table, table_size, cudaMemcpyHostToDevice);
        paillier_cipher_ptr_->agg_tuple<TPI, TPB>(d_cell_table, num_tuples_in_table, max_blocks);
        cudaMemcpy(cell_table, d_cell_table, table_size, cudaMemcpyDeviceToHost);

#ifdef TIME
        timer.start();
#endif
        processResult(cell_table, bin_table, bin_length, rbt, num_rows, num_cols, tuple_length);
#ifdef TIME
        timer.stop();
        std::cout<<"processResult Time "<< timer.duration() <<" US"<<std::endl;
#endif

        //printBinTable(bin_table, bin_length);
        reduce_round += 1;
        
      }

      //printBinTable(bin_table, bin_length);
      // Iterate through the map
      for (auto& pair : bin_table) {
          int key = pair.first;
          CgbnPair hist = pair.second[0];
          CgbnPair* data = (CgbnPair*)malloc(sizeof(CgbnPair));
          *data = hist;
          Buffer buffer((void*)data, mem_size, true);
          result[key] = buffer; // Add the Buffer object to the result map
          
      }

      if (debug_) std::cout << "Finish AddGHPairs" << std::endl;
      if (encrypted_gh_pairs_) {
          clearGHPairs();
      }

      free(cell_table);
      cudaFree(d_cell_table);
      return result;
    }

    std::map<int, Buffer> AddGHPairsOLD(const std::map<int, std::vector<int>>& sample_ids) {
      if (debug_) std::cout << "Calling AddGHPairs with sample_ids size " << sample_ids.size() << std::endl;
      if (!encrypted_gh_pairs_) {
        setGHPairs();
      }
      std::map<int, Buffer> result;

      CgbnPair* d_res_ptr;
      size_t mem_size = sizeof(CgbnPair);
      if (mem_size != 2 * sizeof(cgbn_mem_t<bits>)) {
        std::cout << "Fatal Error" << std::endl;
      }
      ck(cudaMalloc((void **)&d_res_ptr, mem_size));

      if (!paillier_cipher_ptr_->has_pub_key) {
        std::cout << "Can't call AddGHPairs if paillier does not have public key." << std::endl;
        throw std::runtime_error("Can't call AddGHPairs if paillier does not have public key.");
      }

      // Iterate through the map
      for (auto& pair : sample_ids) {
        int key = pair.first;
        const int* sample_id = pair.second.data();
        int count = pair.second.size();

        int* sample_id_d;
        ck(cudaMalloc((void **)&sample_id_d, sizeof(int) * count));
        cudaMemcpy(sample_id_d, sample_id, sizeof(int) * count, cudaMemcpyHostToDevice);

        paillier_cipher_ptr_->sum<TPI,TPB>(d_res_ptr, encrypted_gh_pairs_, sample_id_d, count);

        void* data = malloc(mem_size);
        cudaMemcpy(data, d_res_ptr, mem_size, cudaMemcpyDeviceToHost);
        Buffer buffer(data, mem_size, true);
        result[key] = buffer; // Add the Buffer object to the result map
        cudaFree(sample_id_d);
      }
      cudaFree(d_res_ptr);

      if (debug_) std::cout << "Finish AddGHPairs" << std::endl;
      if (encrypted_gh_pairs_) {
        clearGHPairs();
      }
      return result;
    }
};
} // namespace nvflare

#endif // CUDA_CT_PLUGIN_H
