import grozi_prep as gp
import train


def main():
    gp.etl()
    train.run(imgsz=3264, batch=4, epochs=10, data="grozi.yaml", weights="yolov5l.pt")

if __name__ == "__main__":
    main()
