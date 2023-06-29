def onnx_to_rknn2(onnx_path, platform='rk3588'):
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)

    print('--> config model')
    rknn.config(mean_values=[0, 0, 0], std_values=[255, 255, 255], target_platform=platform)

    # Load onnx model
    print('--> Loading onnx model')
    ret = rknn.load_onnx(model=onnx_path, input_size_list=[[1, 3, 640, 640]])
    if ret != 0:
        print('Load torch model failed!')
        exit(ret)

    # Build model
    print('--> Building rknn model')
    out_path = onnx_path.replace('.onnx', '.rknn')
    ret = rknn.build(do_quantization=False)

    if ret != 0:
        print('Build rknn failed!')
        exit(ret)

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(out_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('Done!')
    rknn.release()
    return out_path


if __name__ == '__main__':
    # ######################## config options ###########################
    project_name = ''
    version = 0


    onnx_path = f'Projects/{project_name}/log/v{version}/train/weights/best.onnx'
    onnx_to_rknn2(onnx_path)