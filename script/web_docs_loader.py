from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import os
import time
import tiktoken
from langchain.docstore.document import Document
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.base import BaseLoader
from scripts.utils import doc_type_classifier,get_original_url_from_directory_path, get_repo_fluffycontext_path, get_repo_dependencies_path, current_script_dir
import pandas as pd

log_file = current_script_dir.parent / 'repo_ingestion.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

class WebDocsLoader(BaseLoader):
    def __init__(
        self,
        path: Union[str, Path],
        git_url: str = None,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        custom_html_tag: Optional[Tuple[str, dict]] = None,
        **kwargs: Optional[Any],
    ):
        """
        Initialize WebDocsLoader

        The loader loops over all files under `path` and extracts the actual content of
        the files by retrieving main html tags. Default main html tags include
        `<main id="main-content>`, <`div role="main>`, and `<article role="main">`. You
        can also define your own html tags by passing custom_html_tag, e.g.
        `("div", "class=main")`. The loader iterates html tags with the order of
        custom html tags (if exists) and default html tags. If any of the tags is not
        empty, the loop will break and retrieve the content out of that tag.

        Args:
            path: The location of pulled readthedocs folder.
            encoding: The encoding with which to open the documents.
            errors: Specify how encoding and decoding errors are to be handled—this
                cannot be used in binary mode.
            custom_html_tag: Optional custom html tag to retrieve the content from
                files.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "Could not import python packages. "
                "Please install it with `pip install beautifulsoup4`. "
            )

        try:
            _ = BeautifulSoup(
                "<html><body>Parser builder library test.</body></html>", **kwargs
            )
        except Exception as e:
            raise ValueError("Parsing kwargs do not appear valid") from e
        
        df_docs_desc = pd.read_csv(f"{get_repo_fluffycontext_path(git_url)}/documentationDescription.csv")
        source_value = df_docs_desc.loc[df_docs_desc['docs_path'] == path, 'sources'].iloc[0]
        

        self.sources = str(source_value)
        self.file_path = Path(path)
        self.file_directory = str(Path(path).parent)
        self.encoding = encoding
        self.errors = errors
        self.custom_html_tag = custom_html_tag
        self.bs_kwargs = kwargs
        self.type = doc_type_classifier(path)
        self.url = get_original_url_from_directory_path(path)
        self.parent_count = 0

    def load(self) -> List[Document]:
        """Load documents."""
        docs = []
        for p in self.file_path.rglob("*"):
            if p.is_dir():
                continue
            with open(p, encoding=self.encoding, errors=self.errors) as f:
                text = self._clean_data(f.read())
                

            tokenizer = tiktoken.get_encoding('cl100k_base')

            def tiktoken_len(text):
                tokens = tokenizer.encode(
                    text,
                    disallowed_special=()
                )
                return len(tokens)

            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,  
            length_function=tiktoken_len,
            separators=['\n\n', '\n', ' ', '']
            )

            relative_path = p.relative_to(self.file_path)
            last_modified_time = os.path.getmtime(p)
            ctime = os.path.getctime(p)
            formatted_last_modified_time = time.strftime('%Y/%m/%d %H:%M', time.localtime(last_modified_time))
            formatted_ctime = time.strftime('%Y/%m/%d %H:%M', time.localtime(ctime))

            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                docs.append(
                    {
                        'text': f"Part {i+1} / {len(chunks)} of documentation \n {chunk}",
                        'metadata': {
                            "sources": str(self.sources),
                            "doc_path": str(p),
                            "last_modified": formatted_last_modified_time,
                            "file_directory": str(self.url),
                            "filename": str(self.url)+str(relative_path),
                            "chunkid": i+1,
                            "total_chunks": len(chunks),
                            "parent": self.parent_count+1,
                            "total_parents": 0,
                            "doc_file_name": str(get_original_url_from_directory_path(str(self.file_path))),
                            "doc_type": str(doc_type_classifier(str(self.file_path)))
                        }
                    }
                )
            
            self.parent_count = self.parent_count + 1

        for doc in docs: 
            doc['metadata']['total_parents'] = self.parent_count + 1

        return docs

    def _clean_data(self, data: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data, **self.bs_kwargs)

        # default tags
        html_tags = [
            ("div", {"role": "main"}),
            ("main", {"id": "main-content"}),
        ]

        if self.custom_html_tag is not None:
            html_tags.append(self.custom_html_tag)

        text = None

        # reversed order. check the custom one first
        for tag, attrs in html_tags[::-1]:
            text = soup.find(tag, attrs)
            # if found, break
            if text is not None:
                break

        if text is not None:
            text = text.get_text()
        else:
            text = ""
        
        if len(text) < 300:
            text = "\n".join([div.get_text() for div in soup.find_all("div")])

        return "\n".join([t for t in text.split("\n") if t])
    
    
def web_docs_loader(git_url):
    exclude_prefix = "archived"
    df = pd.read_csv(f"{get_repo_dependencies_path(git_url)}/documentationList.csv")
    logging.info(f"Loading web docs list:")
    directories = df['Path'].to_list()
    web_docs = []
    for directory in directories:
        doc_type = doc_type_classifier(directory)
        logging.info(f"Type: {doc_type} | Path {directory}")
        if doc_type == 'Web':
            logging.info(f"Loading using web loader")
            web_docs_test = WebDocsLoader(path=directory,git_url=git_url,features="html.parser", errors = "replace")
            web_doc_unit = web_docs_test.load()
            logging.info(len(web_doc_unit))
            web_docs.extend(web_doc_unit)
    return web_docs