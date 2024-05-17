### To implement in ICVV2023:
ICVV2023\
    
    --the rest of the repo ICVV2023 [❌] 
    --demo.py [✅]
    --eyes\
        --data\
            model_weights.pkl [✅]

        eyes_and_mouth_checker.py [✅]
        eye_logger.py [✅] 

---

### To run independent code: 
```bash
python independent_script.py --source_video <path/to/video> --driving_video <path/to/video> --open_mouth 1 --n 100 
```

** --source_video ** path to source video  
** --driving_video ** path to driving video  
** --open_mouth ** int flag for tracking mouth (1 if need opened mouth, 2 if closed, 0 if mouth tracking not required (default 0)  
** --n ** int number of best frames (default 1)  
  
