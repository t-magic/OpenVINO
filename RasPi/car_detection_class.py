import cv2
import numpy as np


# モジュール読み込み 
from openvino.inference_engine import IENetwork, IEPlugin

class RaspiOpenVino:
    def __init__(self, model, weights):

        #model = 'data/model/FP16/R3/person-vehicle-bike-detection-crossroad-0078.xml'
        #weights = 'data/model/FP16/R3/person-vehicle-bike-detection-crossroad-0078.bin'
        # ターゲットデバイスの指定
        plugin = IEPlugin(device="MYRIAD")
        # モデルの読み込み 
        net = IENetwork(model=model, weights=weights)
        #plugin.add_cpu_extension('cpu_extension.dll')
        self.exec_net = plugin.load(network=net)

    def run(self, pict):
        #pict = 'data/cars/cars-on-highway-1.jpg'
        # 入力画像読み込み
        frame = cv2.imread(pict)

        # 入力データフォーマットへ変換 
        img = cv2.resize(frame, (1024, 1024))   # サイズ変更 
        img = img.transpose((2, 0, 1))    # HWC > CHW 
        img = np.expand_dims(img, axis=0) # 次元合せ
        #return frame, img

        # 推論実行 
        out = self.exec_net.infer(inputs={'data': img})
        #return out

        # 出力から必要なデータのみ取り出し 
        out = out['detection_out']
        out = np.squeeze(out) #サイズ1の次元を全て削除

        # 検出されたすべての顔領域に対して１つずつ処理 
        for detection in out:
            # conf値の取得 
            confidence = float(detection[2])

            # バウンディングボックス座標を入力画像のスケールに変換 
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            # conf値が0.5より大きい場合のみバウンディングボックス表示 
            if confidence > 0.1:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
        
        # 画像表示 
        cv2.imshow('frame', frame)
 
        # キーが押されたら終了 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
model = 'data/model/FP16/R3/person-vehicle-bike-detection-crossroad-0078.xml'
weights = 'data/model/FP16/R3/person-vehicle-bike-detection-crossroad-0078.bin'
rov = RaspiOpenVino(model, weights)
'''
pict = 'data/cars/cars-on-highway-1.jpg'
pict = 'data/cars/waymochandlereventjune28-26.jpg'
pict = 'data/cars/GettyImages-88621311.jpg'
rov.run(pict)
'''
