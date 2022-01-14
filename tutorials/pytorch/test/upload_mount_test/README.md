# Test Repo (Temporarily for internal use)

## Uploading and mounting a dataset to Spell for IPU use

### Uploading a dataset

Upload the dataset in the file

```ShellSession
$ spell upload --name 'test_upload' test_file.csv
```

Run the file, mounting the uploaded file from SpellFS:

```ShellSession
$ spell run --machine-type IPUx16 --mount uploads/test_upload/:/mnt/data --pip-req requirements.txt --docker-image graphcore/pytorch:latest "python3 main.py"
```
