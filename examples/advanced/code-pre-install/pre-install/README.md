In this example, we will show how to use the nvflare pre-install install command.

First, you prepare the application_code_zip folder structure:

```bash
application_code_folder
├── application/
│   └── <job_name>/
│               ├── meta.json       # job metadata
│               ├── app_<site>/     # Site custom code
│                  └── custom/      # Site custom code
├── application-share/              # Shared resources
│   └── shared.py
└── requirements.txt       # Python dependencies (optional)
```

We already prepare application-share folder and requirements.txt in this example.
We run the following command to make them a zip folder so we can then distribute to each site:

```bash
python -m zipfile -c application_code.zip application-share requirements.txt
```

And on each site we can then run the following command to pre-install these codes before the NVFlare is up:
```bash
nvflare pre-install install -a application_code.zip -s [site-name] 
```
