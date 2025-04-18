.. _auditing:

Auditing
========
NVFLARE has an auditing mechanism to record events that occur in the system. Both user command events
and critical job events generated by the learning process are recorded.

The audit logging system is primarily implemented in :class:`Auditor<nvflare.fuel.sec.audit.Auditor>`.

Audit File Locations
^^^^^^^^^^^^^^^^^^^^
The audit files are located in different directories depending on the process type:

1. Server Parent (SP) Audit Log:
   - Located in the server's root workspace directory
   - Contains system-level events for the server

2. Server Job (SJ) Audit Log:
   - Located in the job's workspace directory
   - Contains job-specific events for the server

3. Client Parent (CP) Audit Log:
   - Located in the client's root workspace directory
   - Contains system-level events for the client

4. Client Job (CJ) Audit Log:
   - Located in the job's workspace directory
   - Contains job-specific events for the client

To access the audit files for a job, run `download_job <job_id>` and look in the downloaded job's workspace folder
for the SJ and CJ logs (by default named audit.log).

Audit File Format
^^^^^^^^^^^^^^^^^^
The audit file is a text file. Each line in the file is an event. Each event contains headers and an optional message.
Event headers are enclosed in square brackets. The following are some examples of events:

.. code-block::

    [E:b6ac4a2a-eb01-4123-b898-758f20dc028d][T:2022-09-13 13:56:01.280558][U:?][A:_cert_login admin@b.org]
    [E:16392ed4-d6c7-490a-a84b-12685297e912][T:2022-09-1412:59:47.691957][U:trainer@b.org][A:train.deploy]
    [E:636ee230-3534-45a2-9689-d0ec6c90ed45][R:9dbf4179-991b-4d67-be2f-8e4bac1b8eb2][T:2022-09-14 15:08:33.181712][J:c4886aa3-9547-4ba7-902e-eb5e52085bc2][A:train#39027d22-3c70-4438-9c6b-637c380b8669]received task from server

Event Headers
^^^^^^^^^^^^^^^^^^
Event headers specify meta information about the event. Each header is expressed with the header type (one character),
followed by a colon (:) and the value of the header. The following are defined header types and their values.

.. csv-table::
    :header: Header Type,Meaning,Value
    :widths: 5, 10, 20

    E,Event ID,A UUID for the ID of the event
    T,Timestamp,Time of the event
    U,User,Name of the user
    A,Action,User issued command or job's task name and ID
    J,Job,ID of the job
    S,Scope,Name of the job scope
    R,Reference,Reference to peer's event ID
    M,Message,Optional message describing the event

Most of the headers are self-explanatory, except for the R header. Events can be related. For example, a user command
could cause an event to be recorded on both the server and clients. Similarly, a client's action could cause the server
to act on it (e.g. client submitting task results). The R header records the related event ID on the peer. Reference
event IDs can help to correlate events across the system.
