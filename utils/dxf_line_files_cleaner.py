import ezdxf
import numpy as np

def filter_dxf_lines(source_file_path, target_file_path):

    doc = ezdxf.readfile(source_file_path)
    msp = doc.modelspace()

    # Создаём новый документ
    doc_new = ezdxf.new()
    msp_new = doc_new.modelspace()

    count = 0
    for lwpoly in msp.query("LWPOLYLINE"):
        if not lwpoly.is_closed:
            points = np.array(lwpoly.get_points(format="xy"))
            # Добавляем полилинию в новый документ
            msp_new.add_lwpolyline(points)
            count += 1

    # Сохраняем
    doc_new.saveas(target_file_path)
    print(f"Сохранено {count} незамкнутых полилиний в {target_file_path}.dxf")


for idx in range(1, 11):
    filter_dxf_lines(source_file_path=f"../src/base_squares/квадрат_{idx}.dxf",
                     target_file_path=f"../src/base_squares/f_square_{idx}.dxf")
