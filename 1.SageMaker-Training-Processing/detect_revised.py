

FILE = Path(__file__).resolve()
## For SageMaker
# ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = FILE.parents[1]  # YOLOv5 root directory for SageMaker 0 --> 1

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

...........


@torch.no_grad()
def run(
    ..........
):
    
    try:
        ############ MLOps for SageMaker ###########
        w = str(weights[0] if isinstance(weights, list) else weights)
        if w.endswith('.tar.gz'):
            import tarfile
            ap = tarfile.open(w)
            ap.extractall("/opt/ml/processing/weights/")
            ap.close()


        import glob
#         print(glob.glob("/opt/ml/processing/weights/*/*/*"))
        training_job_name = glob.glob("/opt/ml/processing/weights/*/*/*")[0].split("/")[-3]
        LOGGER.info(f" **************** training_job_name : {training_job_name}")
        weights = f"/opt/ml/processing/weights/{training_job_name}/weights/best.pt"
    except:
        print("Not use SageMaker Processinng job")
        pass
    
    source = str(source)
    
    ..........
    
    
    
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        
        ########### SAVE original imgs for SageMaker ###########
        ori_img = im0s.copy() # Add
        
        im = torch.from_numpy(im).to(device)
        
        
        .................

        
        
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    
                    
                    
                    ########### ADD path for SageMaker ###########
                    save_result_path = str(save_dir / 'result_images')  # ADD
                    
                    if not os.path.exists(str(save_result_path)):
                        os.makedirs(str(save_result_path))
                        
                    cv2.imwrite(save_result_path + "/" + p.name, im0) # REVISE
                    ########### ADD path for SageMaker ###########
  

                    
                else:  # 'video' or 'stream'
                    ................
                
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

if __name__ == "__main__":
    opt = parse_opt()
    opt.data = 'data/data_sm.yaml'
    main(opt)