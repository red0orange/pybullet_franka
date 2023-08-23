import cv2
import numpy as np
import textwrap

def display_text_in_window(text, font_scale=1, font_color=(0, 0, 0), line_type=2):
    # 创建一个空白图像
    image = np.ones((600, 800, 3), np.uint8) * 255

    # 使用 textwrap 格式化文字
    wrapper = textwrap.TextWrapper(width=40)
    word_list = wrapper.wrap(text=text)
    new_text = '\n'.join(word_list)

    # 设置文字的位置、字体、大小等
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 将文字添加到图像上
    y0, dy = 50, 30
    for i, line in enumerate(new_text.split('\n')):
        y = y0 + i*dy
        cv2.putText(image, line, (50, y), font, font_scale, font_color, line_type)

    # 显示图像
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用函数
display_text_in_window("""
testtesttesetsete testtesttesetsete testtesttesetsete
""")
