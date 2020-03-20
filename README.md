# Dart-Fss-Classifier

[Dart-Fss](https://github.com/josw123/dart-fss)의 재무제표 추출 성능 향상을 위한 플러그인.

자연어 처리를 이용하여 재무제표 계정과목의 Concept ID 추정함으로써 재무제표 추출 성능 향상. 

## Dependencies

-   [Dart-Fss](https://github.com/josw123/dart-fss) >= 0.3.0
-   [TensorFlow](https://www.tensorflow.org)
-   [KoNLPy](http://konlpy.org/en/latest/)

## Installation

```bash
pip install dart-fss-classifier
```

## Usage

```python

# Dart-fss 라이브러리 불러오기
import dart_fss as dart
# dart_fss_classifier Plugin 불러오기
import dart_fss_classifier

# Attach plugin
assert dart_fss_classifier.attached_plugin() == True

# 회사리스트 불러오기
corp_list = dart.get_corp_list()
# 삼성전자 선택
samsung = corp_list.find_by_corp_name(corp_name='삼성전자', exactly=True)[0]
# 재무제표 추출
fs = samsung.extract_fs(bgn_de='20100101')
```