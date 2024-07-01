from data_provider.data_loader import Dataset_ETT_hour,Dataset_ETT_hour_decomposed, Dataset_ETT_minute,Dataset_ETT_minute_decomposed,Dataset_Custom



from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh1_decomposed': Dataset_ETT_hour_decomposed,
    'ETTh2': Dataset_ETT_hour,
    'ETTh2_decomposed': Dataset_ETT_hour_decomposed,
    'ETTm1': Dataset_ETT_minute,
    'ETTm1_decomposed': Dataset_ETT_minute_decomposed,
    'ETTm2': Dataset_ETT_minute,
    'ETTm2_decomposed': Dataset_ETT_minute_decomposed,
    'ECL': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom
   
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            percent=percent
        )
        # batch_size = args.batch_size
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
