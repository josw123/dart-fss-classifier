import re
import math

from pandas import DataFrame
from typing import Tuple, List, Union
from dart_fss.fs.extract import find_all_columns, extract_account_title

from dart_fss_classifier.classifier import guess


def attached_plugin():
    """ dart-fss 라이브러리에 플로그인 연결

    :return:
    bool
        정상적으로 연결시 True / 오류 발생시 False
    """
    try:
        from dart_fss.fs.extract import additional_comparison_function
        additional_comparison_function.append(compare_df_and_ndf_cnn)
        return True
    except BaseException:
        return False


def guess_concept_id(text):
    """
    Concept_id 추측

    Parameters
    ----------
    text: str
        추측할 Label

    Returns
    -------
    str
        추측된 Concept_id
    """
    return guess(text)


def compare_df_and_ndf_cnn(column: Tuple[Union[str, Tuple[str]]],
                             df: DataFrame, ndf: DataFrame, ldf: DataFrame,
                             ndata: List[Union[float, str, None]],
                             nlabels: List[str]) -> Tuple[List[Union[float, str]], List[str]]:
    """
    Convolutional neural network 를 시용하여 데이터를 검색하는 함수

    Parameters
    ----------
    column: tuple
        추가할 column Name
    df: dict of { str: DataFrame }
        데이터를 추가할 DataFrame, 추출된 결과값이 누적된 DataFrame
    ndf: dict of { str: DataFrame }
        데이터를 검색할 DataFrame, Report에서 추출한 새로운 DataFrame
    ndata: list of float
        추가할 column의 데이터 리스트
    nlabels: list of str
        추가할 column의 label 리스트

    Returns
    -------
    tuple of list
        추가할 column의 데이터 리스트, 추가할 column의 label 리스트
    """
    # CNN 처리시 사용
    concept_none_data = {}
    df_label_column = find_all_columns(df, 'label_ko')[0]

    is_concept = True
    df_concept_column = find_all_columns(df, 'concept_id')
    if len(df_concept_column) == 0:
        is_concept = False
    else:
        df_concept_column = df_concept_column[0]

    ndf_label_column = find_all_columns(ndf, 'label_ko')[0]

    for idx, value in enumerate(ndata):
        if isinstance(value, str):
            pass
        elif value is None:
            pass
        elif math.isnan(value):
            pass
        else:
            continue

        label = df[df_label_column].iloc[idx]
        label = re.sub(r'\s+', '', label)
        label = extract_account_title(label)

        if is_concept:
            concept_id = df[df_concept_column].iloc[idx]
        else:
            concept_id = guess_concept_id(label)

        if concept_id is not None:
            concept_none_data[concept_id] = idx

    matched = []
    used = []
    for idx in range(len(ndf)):
        if idx in matched:
            continue
        label = extract_account_title(ndf[ndf_label_column].iloc[idx])
        concept_id = guess_concept_id(label)
        index = concept_none_data.get(concept_id)
        if index is not None and index not in used:
            value = ndf[column].iloc[idx]
            if isinstance(value, str):
                pass
            else:
                used.append(index)
                matched.append(idx)
                ndata[index] = value
                nlabels[index] = label

    return ndata, nlabels
