package top.aias.ocr.utils.recognition;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.opencv.OpenCVImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import top.aias.ocr.utils.common.ImageInfo;
import top.aias.ocr.utils.common.RotatedBox;
import top.aias.ocr.utils.detection.OCRDetectionTranslator;
import top.aias.ocr.utils.opencv.NDArrayUtils;
import top.aias.ocr.utils.opencv.OpenCVUtils;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
/**
 * 多线程文字识别
 *
 * @author Calvin
 * @mail 179209347@qq.com
 * @website www.aias.top
 */
public final class OcrV3MultiThreadRecognition {

    private static final Logger logger = LoggerFactory.getLogger(OcrV3MultiThreadRecognition.class);

    public OcrV3MultiThreadRecognition() {
    }

    /**
     * 图片推理
     *
     * @param manager
     * @param image
     * @param recognitionModel
     * @param detector
     * @param threadNum
     * @return
     * @throws TranslateException
     */
    /**
     * 使用给定的检测模型和识别模型，对输入图像进行物体检测和识别，返回旋转框的结果列表。
     *
     * @param manager NDManager，用于管理计算资源和内存。
     * @param image 输入的图像对象。
     * @param recognitionModel 识别模型，用于对检测到的物体进行识别。
     * @param detector 检测模型，用于检测图像中的物体并返回其边界框。
     * @param threadNum 使用的线程数量，用于并行处理图像识别。
     * @return 返回一个包含识别结果的旋转框列表。
     * @throws TranslateException 如果处理过程中发生错误，则抛出异常。
     */
    public List<RotatedBox> predict(
            NDManager manager, Image image, ZooModel recognitionModel, Predictor<Image, NDList> detector, int threadNum)
            throws TranslateException {
        // 使用检测模型预测图像中的物体边界框
        NDList boxes = detector.predict(image);
        // 将预测结果绑定到NDManager上，自动管理内存
        boxes.attach(manager);

        // 创建一个并发队列，用于存储图像信息
        ConcurrentLinkedQueue<ImageInfo> queue = new ConcurrentLinkedQueue<>();
        for (int i = 0; i < boxes.size(); i++) {

            NDArray box = boxes.get(i);

            // 从边界框数组中提取四个点的坐标
            float[] pointsArr = box.toFloatArray();
            float[] lt = java.util.Arrays.copyOfRange(pointsArr, 0, 2);
            float[] rt = java.util.Arrays.copyOfRange(pointsArr, 2, 4);
            float[] rb = java.util.Arrays.copyOfRange(pointsArr, 4, 6);
            float[] lb = java.util.Arrays.copyOfRange(pointsArr, 6, 8);

            // 计算裁剪图像的宽高
            int img_crop_width = (int) Math.max(distance(lt, rt), distance(rb, lb));
            int img_crop_height = (int) Math.max(distance(lt, lb), distance(rt, rb));

            // 计算图像的四个裁剪点
            List<Point> srcPoints = new ArrayList<>();
            srcPoints.add(new Point(lt[0], lt[1]));
            srcPoints.add(new Point(rt[0], rt[1]));
            srcPoints.add(new Point(rb[0], rb[1]));
            srcPoints.add(new Point(lb[0], lb[1]));
            List<Point> dstPoints = new ArrayList<>();
            dstPoints.add(new Point(0, 0));
            dstPoints.add(new Point(img_crop_width, 0));
            dstPoints.add(new Point(img_crop_width, img_crop_height));
            dstPoints.add(new Point(0, img_crop_height));

            // 进行透视变换，将图像裁剪成正方形
            Mat srcPoint2f = NDArrayUtils.toMat(srcPoints);
            Mat dstPoint2f = NDArrayUtils.toMat(dstPoints);
            Mat cvMat = OpenCVUtils.perspectiveTransform((Mat) image.getWrappedImage(), srcPoint2f, dstPoint2f);
            Image subImg = OpenCVImageFactory.getInstance().fromImage(cvMat);

            // 调整图像方向，以适应高度大于宽度的情况
            subImg = subImg.getSubImage(0, 0, img_crop_width, img_crop_height);
            if (subImg.getHeight() * 1.0 / subImg.getWidth() > 1.5) {
                subImg = rotateImg(manager, subImg);
            }

            // 创建图像信息对象，并加入队列
            ImageInfo imageInfo = new ImageInfo(subImg, box);
            queue.add(imageInfo);
        }

        // 使用线程池并行处理图像识别
        List<InferCallable> callables = new ArrayList<>(threadNum);
        for (int i = 0; i < threadNum; i++) {
            callables.add(new InferCallable(recognitionModel, queue));
        }

        ExecutorService es = Executors.newFixedThreadPool(threadNum);
        List<ImageInfo> resultList = new ArrayList<>();
        try {
            // 提交所有任务并收集结果
            List<Future<List<ImageInfo>>> futures = new ArrayList<>();
            for (InferCallable callable : callables) {
                futures.add(es.submit(callable));
            }

            // 合并所有子任务的结果
            for (Future<List<ImageInfo>> future : futures) {
                List<ImageInfo> subList = future.get();
                if (subList != null) {
                    resultList.addAll(subList);
                }
            }

            // 关闭所有可关闭的资源
            for (InferCallable callable : callables) {
                callable.close();
            }
        } catch (InterruptedException | ExecutionException e) {
            logger.error("", e);
        } finally {
            es.shutdown();
        }

        // 构建并返回旋转框结果列表
        List<RotatedBox> rotatedBoxes = new ArrayList<>();
        for (ImageInfo imageInfo : resultList) {
            RotatedBox rotatedBox = new RotatedBox(imageInfo.getBox(), imageInfo.getName());
            rotatedBoxes.add(rotatedBox);

            // 释放图像资源
            Mat wrappedImage = (Mat) imageInfo.getImage().getWrappedImage();
            wrappedImage.release();
        }

        return rotatedBoxes;
    }



    /**
     * 中文检测模型
     *
     * @return
     */
    public Criteria<Image, NDList> detectCriteria() {
        Criteria<Image, NDList> criteria =
                Criteria.builder()
                        .optEngine("OnnxRuntime")
                        .optModelName("inference")
                        .setTypes(Image.class, NDList.class)
                        .optModelPath(Paths.get("1_image_sdks/ocr_sdks/ocr_v3_sdk/models/ch_PP-OCRv3_det_infer_onnx.zip"))
                        .optTranslator(new OCRDetectionTranslator(new ConcurrentHashMap<String, String>()))
                        .optProgress(new ProgressBar())
                        .build();

        return criteria;
    }

    /**
     * 中文识别模型
     *
     * @return
     */
    public Criteria<Image, String> recognizeCriteria() {
        ConcurrentHashMap<String, String> hashMap = new ConcurrentHashMap<>();

        Criteria<Image, String> criteria =
                Criteria.builder()
                        .optEngine("OnnxRuntime")
                        .optModelName("inference")
                        .setTypes(Image.class, String.class)
                        .optModelPath(Paths.get("1_image_sdks/ocr_sdks/ocr_v3_sdk/models/ch_PP-OCRv3_rec_infer_onnx.zip"))
                        .optProgress(new ProgressBar())
                        .optTranslator(new PpWordRecTranslator(hashMap))
                        .build();

        return criteria;
    }

    private static class InferCallable implements Callable<List<ImageInfo>> {
        private Predictor<Image, String> recognizer;
        private ConcurrentLinkedQueue<ImageInfo> queue;
        private List<ImageInfo> resultList = new ArrayList<>();

        public InferCallable(ZooModel recognitionModel, ConcurrentLinkedQueue<ImageInfo> queue) {
            recognizer = recognitionModel.newPredictor();
            this.queue = queue;
        }

        public List<ImageInfo> call() {
            ImageInfo imageInfo = queue.poll();
            try {
                while (imageInfo != null) {
                    String name = recognizer.predict(imageInfo.getImage());
                    imageInfo.setName(name);
                    imageInfo.setProb(-1.0);
                    resultList.add(imageInfo);
                    imageInfo = queue.poll();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            return resultList;
        }

        public void close() {
            recognizer.close();
        }
    }

    /**
     * 欧式距离计算
     *
     * @param point1
     * @param point2
     * @return
     */
    private float distance(float[] point1, float[] point2) {
        float disX = point1[0] - point2[0];
        float disY = point1[1] - point2[1];
        float dis = (float) Math.sqrt(disX * disX + disY * disY);
        return dis;
    }

    /**
     * 图片旋转
     *
     * @param manager
     * @param image
     * @return
     */
    private Image rotateImg(NDManager manager, Image image) {
        NDArray rotated = NDImageUtils.rotate90(image.toNDArray(manager), 1);
        return ImageFactory.getInstance().fromNDArray(rotated);
    }
}
