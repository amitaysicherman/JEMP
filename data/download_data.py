import os
import requests
from tqdm import tqdm
import gzip
import shutil
from dataclasses import dataclass


@dataclass
class DataSource:
    url: str
    file_name: str
    base_dir: str

    def get_file_path(self):
        return os.path.join(self.base_dir, self.file_name)


ALL_DATASETS = {
    "pubchem": DataSource(
        url="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz",
        file_name="CID-SMILES.gz",
        base_dir="pubchem",
    ),
    "USPTO": DataSource(
        url="https://az.app.box.com/index.php?rm=box_download_shared_file&shared_name=7eci3nd9vy0xplqniitpk02rbg9q2zcq&file_id=f_854847813119",
        file_name="data.pickle",
        base_dir="USPTO",
    ),
    "PaRoutes": DataSource(
        url="https://zenodo.org/record/7341155/files/all_loaded_routes.json.gz?download=1",
        file_name="all_loaded_routes.json.gz",
        base_dir="PaRoutes",
    )
}


def download_url(data_source: DataSource):
    if not os.path.exists(data_source.base_dir):
        os.makedirs(data_source.base_dir)
    filename = data_source.get_file_path()
    if not os.path.exists(filename) and not os.path.exists(filename.replace(".gz", "")):
        with requests.get(data_source.url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            pbar = tqdm(
                total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True
            )
            with open(filename, "wb") as fileobj:
                for chunk in response.iter_content(chunk_size=1024):
                    fileobj.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()


def extract_zip(zip_file):
    name_out = zip_file.replace(".gz", "")
    if os.path.exists(name_out):
        return
    with gzip.open(zip_file, 'rb') as f_in:
        with open(name_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # remove the original file
    os.remove(zip_file)


def download_all():
    download_url(ALL_DATASETS["pubchem"])
    extract_zip(ALL_DATASETS["pubchem"].get_file_path())
    download_url(ALL_DATASETS["USPTO"])
    download_url(ALL_DATASETS["PaRoutes"])
    extract_zip(ALL_DATASETS["PaRoutes"].get_file_path())


if __name__ == "__main__":
    download_all()
    print("All datasets downloaded and extracted")
