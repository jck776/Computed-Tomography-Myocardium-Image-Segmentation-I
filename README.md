# Computed-Tomography-Myocardium-Image-Segmentation-I
AI CUP 2025 Fall Competition - Computed Tomography Myocardium Image Segmentation I - Myocardium Image Segmentation

總結而言，本次專案主要在CardiacSegV2程式基礎上，透過優化 swinunetr 模型的訓練超參數並結合多模型的集成學習策略，將模型泛化能力推至競賽的前標水平。

---

## 使用Jupyter notebook執行訓練與推論
* [Computed-Tomography-Myocardium-Image-Segmentation-I.ipynb](https://github.com/jck776/Computed-Tomography-Myocardium-Image-Segmentation-I/blob/main/Computed-Tomography-Myocardium-Image-Segmentation-I.ipynb)

## 環境建立
* 請依照 [baseline colab教學完成環境設定與資料集下載](https://colab.research.google.com/drive/1iC7i_EWCZsCr5T-7jDD77V8dt_simGsn?usp=sharing)

## 訓練集與JSON設定檔位置

* 將訓練資料(CT image, label)放置CardiacSegV2/dataset/chgh下

<img width="300" height="430" alt="image" src="https://github.com/user-attachments/assets/82e910b4-bef0-426a-b5be-236140c034af" />


* 將資料集設定檔放置在CardiacSegV2/exps/data_dicts/chgh下

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0a0f9adc-0cb9-4eca-b388-31f79cb7239b" />

---
* [簡報_構想說明](https://github.com/jck776/Computed-Tomography-Myocardium-Image-Segmentation-I/blob/main/doc/TEAM_7987_投影片_心臟CT分割策略.pdf)
* [8個優選模型檔(權重、訓練紀錄)](https://zenodo.org/records/17840566)
