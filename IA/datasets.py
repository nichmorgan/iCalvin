# -*- coding: utf-8 -*-
# from flaskr.settings import CORPUS_DATABASE_PATH, CORPUS_DOCUMENTS_DIR
import logging
import os
from io import BytesIO

import nltk
import numpy as np
import pandas as pd
import requests
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.pdfpage import PDFPage
import re

CORPUS_DATABASE_PATH = r'./data/dataset.csv'
CORPUS_DOCUMENTS_DIR = r'./documents'

log = logging.getLogger('app.datasets')

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')


class Dataset:
    _dataset_config = {
        'breve_catecismo_westminster': {
            'pt-br': {
                'url': 'https://ipb.org.br/uploads/breve-catecismo.pdf',
                'nome': 'Breve Catecismo de Westminster',
                'regex': {
                    'perguntas': 'pergunta',
                    'respostas': 'r.',
                    'referencias': 'ref',
                    'fim_documento': 'os dez mandamentos'
                }
            }
        }
    }
    _dataset = pd.DataFrame(columns=['documento', 'n', 'pergunta', 'resposta', 'referencias'])

    def carregar_breve_catecismo_westminster(self, idioma='pt-br'):
        log.debug(f'Buscando breve catecismo em "{idioma}".')
        if idioma in self._dataset_config['breve_catecismo_westminster']:
            log.debug(f'Idioma "{idioma}" encontrado.')
            config = self._dataset_config['breve_catecismo_westminster'][idioma]
            regex = config['regex']

            file_dir = os.path.abspath(CORPUS_DOCUMENTS_DIR)
            os.makedirs(file_dir, exist_ok=True)

            file_name = (config['nome'] + '.pdf').strip()
            file_path = os.path.join(file_dir, file_name)
            self._download_pdf_and_save(config['url'], file_path)
            texto = self._convert_pdf_to_text(file_path)
            tokens = pd.Series(nltk.line_tokenize(texto)).str.strip()

            log.debug(f'Iniciando extração de campos.')
            perguntas = tokens[tokens.str.lower().str.strip().str.startswith(regex['perguntas'])] \
                .copy()
            respostas = tokens[tokens.str.lower().str.strip().str.startswith(regex['respostas'])] \
                .copy()
            referencias = tokens[tokens.str.lower().str.strip().str.startswith(regex['referencias'])] \
                .copy()
            fim_documento = tokens[tokens.str.lower().str.strip().str.startswith(regex['fim_documento'])] \
                .copy()

            log.debug(f'Extração finalizada.')
            if (perguntas.shape[0] == respostas.shape[0] == referencias.shape[0]):
                log.debug(f'Iniciando correção dos dados.')
                for idx in range(perguntas.shape[0]):
                    perguntas[perguntas.index[idx]] = ' '.join(tokens[perguntas.index[idx]:respostas.index[idx]].values)
                    perguntas[perguntas.index[idx]] = re.sub(r'  ', ' ', perguntas[perguntas.index[idx]]).strip().split(' ', 2)[-1]

                    respostas[respostas.index[idx]] = ' '.join(tokens[respostas.index[idx]:referencias.index[idx]].values)
                    respostas[respostas.index[idx]] = re.sub(r'  ', ' ',respostas[respostas.index[idx]]).strip().split(' ', 1)[-1]

                    if idx + 1 < perguntas.shape[0]:
                        referencias[referencias.index[idx]] = ' '.join(tokens[referencias.index[idx]:perguntas.index[idx + 1]].values)
                        referencias[referencias.index[idx]] = re.sub(r'  ', ' ',referencias[referencias.index[idx]]).strip().split(' ', 1)[-1]
                    else:
                        referencias[referencias.index[idx]] = ' '.join(tokens[referencias.index[idx]:fim_documento.index[0]].values)
                        referencias[referencias.index[idx]] = re.sub(r'  ', ' ',referencias[referencias.index[idx]]).strip().split(' ', 1)[-1]

                perguntas = perguntas.replace('', np.nan).dropna()
                respostas = respostas.replace('', np.nan).dropna()
                referencias = referencias.replace('', np.nan).dropna()

                log.debug(f'Incorporando documentos ao dataset.')
                self._incorporar_documento_ao_dataset(config['nome'], perguntas, respostas, referencias)

            else:
                log.warning('Tamanhos desiguais encontrados durante a extração.')

        else:
            log.warning(f'O idioma "{idioma}" não foi encontrado.')

    def _download_pdf_and_save(self, url, path2save):
        if not os.path.exists(path2save):
            log.debug(f'Documento não encontrado, será baixado em: {url}')
            request = requests.get(url, stream=True)
            with open(path2save, 'wb') as fpdf:
                fpdf.write(request.content)
                fpdf.close()
            log.debug(f'Documento salvo em: {path2save}')
        else:
            log.debug(f'O arquivo {os.path.basename(os.path.abspath(path2save))} já existe.')

    def _convert_pdf_to_text(self, fpath):
        rsrcmgr = PDFResourceManager()
        retstr = BytesIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(fpath, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()
        text = text.decode('utf-8')

        fp.close()
        device.close()
        retstr.close()
        return text

    def _incorporar_documento_ao_dataset(self, documento, perguntas, respostas, referencias):
        if len(perguntas) == len(respostas) == len(referencias):
            documento = pd.Series([documento] * len(perguntas))
            n = pd.Series(range(1, len(perguntas) + 1))
            dataset = pd.DataFrame(
                {self._dataset.columns[i]: dados.values for i, dados in
                 enumerate([documento, n, perguntas, respostas, referencias])}
                , index=range(len(perguntas)))

            self._dataset = self._dataset.append(dataset, ignore_index=True)

            dataset_dir = os.path.abspath(os.path.dirname(CORPUS_DATABASE_PATH))
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_path = os.path.abspath(CORPUS_DATABASE_PATH)
            self._dataset.to_csv(dataset_path, index=False)

        else:
            print('Tamanhos desiguais.')


log.debug('Iniciando teste.')
Dataset().carregar_breve_catecismo_westminster()
