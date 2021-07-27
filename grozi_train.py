import grozi_prep as gp
import train


def main():
    gp.etl()
    train.run(imgsz=408, batch=2, epochs=10, data="grozi.yaml", weights="yolov5m.pt")

if __name__ == "__main__":
    main()
