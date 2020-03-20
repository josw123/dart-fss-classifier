# -*- coding: utf-8 -*-
import json, os
import numpy as np
import tensorflow as tf
import pkg_resources

from typing import List
from konlpy.tag import Kkma

from dart_fss.filings.reports import Report
from dart_fss.utils import Singleton, create_folder, is_notebook
from dart_fss.fs.extract import analyze_xbrl, find_all_columns


def guess(text):
    """
    Concept_id 추측 함수

    Parameters
    ----------
    text: str
        추측할 문장

    Returns
    -------
    str
        추측된 Concept_id

    """
    c = Classifier()
    return c.guess(text)


def load_dataset_and_ccn_model(**kwargs):
    """
    Lookup Table 및 CNN Model 로딩

    Other Parameters
    ----------
    path: str
        데이터 로딩 위치

    Returns
    -------

    """
    path = kwargs.get('path', None)
    c = Classifier()
    c.load(path)


def generate_default_dataset_and_cnn_model(reports_no: int = 25, units: int = 256,
                                           dropout: float = 0.2, epochs: int = 50,
                                           batch_size: int = 512):
    """
    기본설정을 이용한 Dataset 및 CNN Model 생성

    Parameters
    ----------
    reports_no: int
        사용할 Report 수(Max: 100)
    units: int
        Dense Layer의 unit 수
    dropout: float
        dropout rate
    epochs: int
        학습 반복 횟수
    batch_size: int
        batch_size 수
    """
    from dart_fss.filings import search
    from datetime import datetime

    now = datetime.now()
    end_dt = datetime(year=now.year, month=7, day=31).strftime('%Y%m%d')
    start_dt = datetime(year=now.year, month=5, day=1).strftime('%Y%m%d')
    reports = search(bgn_de=start_dt, end_de=end_dt,pblntf_detail_ty='a001', page_count=reports_no)
    generate_dataset_and_cnn_model(reports=reports, units=units, dropout=dropout, epochs=epochs, batch_size=batch_size)


def generate_dataset_and_cnn_model(**kwargs):
    """
    Dataset 및 CNN Model 생성을 위한 Method

    Other Parameters
    ----------------
    reports: list of Report
        데이터를 추출할 Report 리스트
    units: int
        Dense Layer의 unit 수
    dropout: float
        dropout rate
    epochs: int
        학습 반복 횟수
    batch_size: int
        batch_size 수
    path: str
        모델 및 Lookup Table 을 저장할 위치
    """
    reports = kwargs['reports']
    units = kwargs.get('units', 256)
    dropout = kwargs.get('dropout', 0.2)
    epochs = kwargs.get('epochs', 50)
    batch_size = kwargs.get('batch_size', 512)
    path = kwargs.get('path', None)

    c = Classifier()
    c.gen_dataset(reports)
    c.gen_model(units=units, dropout=dropout, epochs=epochs, batch_size=batch_size)
    c.save(path)


class Classifier(metaclass=Singleton):
    """
     Convolutional Neural Networks 모델을 이용하여 Label의 Concept_id를 추측하는 클래스

    Attributes
    ----------
    konlpy
        형태소 추출기 Hannanum, Kkma, Komoran, Mecab, Okt 설정 가능
        자세한 사항은 http://konlpy.org/ 참고
    word_dict
        단어 Lookup Table
    concept_dict
        concept_id Lookup Table
    model
        CNN 모델
    is_load
        CNN 모델 및 Lookup Table 로딩 여부
    """
    def __init__(self):
        self.konlpy = Kkma()
        self._dataset = None
        self.word_dict = None
        self.concept_dict = None
        self._x_train = None
        self._y_train = None
        self.model = None
        self.is_load = False

    def extract_nouns(self, text):
        """
        KoNLPy을 이용하여 명사만 추출

        Parameters
        ----------
        text: str
            추출할 문장

        Returns
        -------
        list of str
            추출된 명사 리스트

        """
        return self.konlpy.nouns(text)

    def gen_dataset(self, reports):
        """

        Report들에서 XBRL 파일을 추출후 Concept_id와 Label 값을 이용하여 CNN 모델 학습

        Parameters
        ----------
        reports: list of Report
            추출할 Report 리스트
        """
        self._extract_dataset(reports)
        self._gen_word_dict()
        self._gen_concept_dict()
        self._gen_x_train()
        self._gen_y_train()
        self.is_load = True

    def _extract_dataset(self, reports: List[Report]):
        """
        Report에 포함된 XBRL 파일에서 Concept_id 와 Label 값 추출

        Parameters
        ----------
        reports: list of Report
            추출할 Report 리스트
        """
        if is_notebook():
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm

        dataset = []
        for report in tqdm(reports, desc='Extracting concept_id and label_ko', unit='report'):
            df_fs = analyze_xbrl(report)
            if df_fs is None:
                continue
            for tp in df_fs:
                df = df_fs[tp]
                if df is not None:
                    concept_column = find_all_columns(df, 'concept_id')[0]
                    label_ko_column = find_all_columns(df, 'label_ko')[0]
                    for idx in range(len(df)):
                        concept_id = df[concept_column].iloc[idx]
                        label_ko = df[label_ko_column].iloc[idx]
                        if concept_id and label_ko:
                            try:
                                label = self.extract_nouns(label_ko)
                                dataset.append((concept_id, label))
                            except BaseException:
                                continue

        self._dataset = dataset

    def _gen_word_dict(self):
        """ 단어 Lookup Table 생성 """
        word_index = dict()
        for _, nouns in self._dataset:
            for noun in nouns:
                if word_index.get(noun) is None:
                    word_index[noun] = 0
                word_index[noun] += 1

        word_dict = dict()
        for idx, (noun, _) in enumerate(sorted(word_index.items(), key=lambda x: x[1], reverse=True)):
            word_dict[noun] = idx + 1

        self.word_dict = word_dict

    def _gen_concept_dict(self):
        """ concept_id Lookup Table 생성 """
        concepts = set()
        for concept, _ in self._dataset:
            concepts.add(concept)

        concept_dict = dict()
        for idx, concept in enumerate(concepts):
            concept_dict[concept] = idx + 1
        self.concept_dict = concept_dict

    def _gen_x_train(self):
        """ 입력값 변환 """
        dataset = []
        for concept_id, label_ko in self._dataset:
            dataset.append([self.word_dict[x] for x in label_ko])
        x_train = self.vectorize_sequences(dataset)
        self._x_train = x_train

    def _gen_y_train(self):
        """ 결과값 변환 """
        dataset = [self.concept_dict[concept] for concept, _ in self._dataset]
        y_train = tf.keras.utils.to_categorical(dataset)
        self._y_train = y_train

    @property
    def input_length(self):
        return len(self.word_dict) + 1

    @property
    def output_length(self):
        return len(self.concept_dict) + 1

    def gen_model(self, units: int = 256, dropout: float = 0.2, epochs: int = 50, batch_size: int = 512):
        """
        Keras를 이용한 CNN 모델 생성 및 학습

        Parameters
        ----------
        units: int
            unit 수
        dropout: float
            dropout rate
        epochs: int
            학습 반복 횟수
        batch_size: int
             batch_size 수
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units, activation='relu', input_shape=(self.input_length,)),
            tf.keras.layers.Dropout(rate=dropout),
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dense(self.output_length, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        length = int(len(self._x_train) / 5)

        x_val = self._x_train[:length * 3]
        partial_x_train = self._x_train[length*3:length*4]
        x_test = self._x_train[length*4:]

        y_val = self._y_train[:length * 3]
        partial_y_train = self._y_train[length*3:length*4]
        y_test = self._y_train[length*4:]

        print("\n==========Model Fit==========\n")
        model.fit(x_val, y_val, epochs=epochs, batch_size=batch_size, validation_data=(partial_x_train, partial_y_train))
        print("\n==========Model Evaluation==========\n")
        model.evaluate(x_test, y_test)
        self.model = model

    def vectorize_sequences(self, sequences: List[List[str]]) -> List[List[int]]:
        """ Label에 포함된 단어를 0과 1의 리스트로 변환"""
        results = np.zeros((len(sequences), self.input_length))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    def save(self, path=None):
        """
        Convolutional Neural Networks 모델 및 Dictionary 저장

        Parameters
        ----------
        path: str
            데이터 저장 위치
        """
        if path is None:
            path = pkg_resources.resource_filename('dart_fss_classifier', 'data/')
        create_folder(path)
        file = os.path.join(path, 'dict.json')

        config = {
            'word_dict': self.word_dict,
            'concept_dict': self.concept_dict,
        }
        model_file = os.path.join(path, 'model.h5')
        self.model.save(model_file)
        with open(file, 'w') as outfile:
            json.dump(config, outfile)

    def load(self, path: str = None) -> str:
        """
        Convolutional Neural Networks 모델 및 Dictionary 로딩

        Parameters
        ----------
        path: str
            데이터 위치
        """
        if path is None:
            path = pkg_resources.resource_filename('dart_fss_classifier', 'data/')
        file = os.path.join(path, 'dict.json')
        if not os.path.isfile(file):
            raise FileExistsError("The dictionary does not exist. Please run 'generate_default_dataset_and_cnn_model'.")

        model_file = os.path.join(path, 'model.h5')
        if not os.path.isfile(model_file):
            raise FileExistsError("The Keras model does not exist. Please run 'generate_default_dataset_and_cnn_model'.")

        self.model = tf.keras.models.load_model(model_file)
        with open(file) as json_file:
            data = json.load(json_file)
            self.word_dict = data['word_dict']
            self.concept_dict = data['concept_dict']

        self.is_load = True

    def guess(self, text: str) -> str:
        """
        Concept_id 추측 Method

        Parameters
        ----------
        text: str
            Label 명

        Returns
        -------
        str
            추측한 Concept_id

        """
        if not self.is_load:
            self.load()
        data = []
        for noun in self.extract_nouns(text):
            try:
                word = self.word_dict[noun]
                data.append(word)
            except BaseException:
                pass

        d = self.vectorize_sequences([data])
        prediction = np.argmax(self.model.predict(d))
        for key, value in self.concept_dict.items():
            if value == prediction:
                return key
        return None
