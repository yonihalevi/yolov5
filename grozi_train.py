import grozi_prep as gp
import train


def main():
    gp.etl()
    train.run(imgsz=816, batch=4, epochs=100, data="grozi.yaml", weights="yolov5m.pt")

if __name__ == "__main__":
    main()
