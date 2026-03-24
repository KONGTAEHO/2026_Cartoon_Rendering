import cv2
import numpy as np

def cartoonify_image(image_path, output_path):
    # 1. 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print("Image load failed.")
        return

    # 2. 색감 보정 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.3  # 채도 30% 증가
    hsv[:, :, 2] *= 1.15 # 명도 15% 증가
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    color_boost = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 일러스트 특징 추가
    color_boost = color_boost.astype(np.float32)
    color_boost[:, :, 2] += 15 # Red 채널 증가 (핑크빛 생기)
    color_boost[:, :, 1] += 5  # Green 채널 살짝 증가
    color_boost = np.clip(color_boost, 0, 255).astype(np.uint8)

    # 3. 피부/배경 스무딩 (에어브러시 느낌)
    # 경계선을 유지하면서 내부 질감을 수채화처럼 부드럽게 뭉개기
    smooth = color_boost.copy()
    for _ in range(4):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. 연하고 부드러운 선화 (가벼운 펜/연필 느낌)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    # 선이 너무 진하고 거칠어지지 않게 부드럽게 추출
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    
    # 투박한 순흑색 선이 아니라, 그림과 어우러지는 부드러운 멀티플라이 합성
    edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_float = edges_3c.astype(np.float32) / 255.0
    styled = cv2.multiply(smooth.astype(np.float32), edges_float)
    styled = np.clip(styled, 0, 255).astype(np.uint8)

    # 5. 뽀샤시 / 글로우 효과 - Pixiv 일러스트 느낌
    # 블러 처리된 이미지를 겹쳐 은은하게 발광(Glow)하는 효과 추가
    glow_blur = cv2.GaussianBlur(styled, (25, 25), 0)
    # 가산 혼합(addWeighted)으로 화사하게 빛나는 느낌 내기
    final_out = cv2.addWeighted(styled, 0.75, glow_blur, 0.4, 0)

    # 6. 결과 이미지 저장 및 화면 출력
    cv2.imwrite(output_path, final_out)
    print(f"Pixiv style image saved to: {output_path}")

    # 화면에서 바로 확인하기
    # h, w = final_out.shape[:2]
    # target_height = 600
    # target_width = int(w * (target_height / h))
    # cv2.imshow("Pixiv Style", cv2.resize(final_out, (target_width, target_height)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# image1, image2 입력 및 cartoon_image1, 2 출력
for i in range(1, 3):
    input_image = f'image{i}.jpg'
    output_image = f'cartoon_image{i}.jpg'
    print(f"Processing {input_image}...")
    cartoonify_image(input_image, output_image)