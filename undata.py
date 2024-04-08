# 未正規化的資料
train_data_unnorm = pd.read_csv(folder + '/train.csv')
x_unnorm = train_data_unnorm.iloc[:, 1:].values  # 未進行除以255的操作
y_unnorm = train_data_unnorm.iloc[:, 0].values

# 拆分未正規化的資料
train_x_unnorm, val_x_unnorm, train_y_unnorm, val_y_unnorm = train_test_split(
    x_unnorm, y_unnorm, test_size=0.2)

# 轉換為 Tensor
train_x_unnorm = torch.from_numpy(train_x_unnorm).type(torch.FloatTensor)
train_y_unnorm = torch.from_numpy(train_y_unnorm).type(torch.LongTensor)
val_x_unnorm = torch.from_numpy(val_x_unnorm).type(torch.FloatTensor)
val_y_unnorm = torch.from_numpy(val_y_unnorm).type(torch.LongTensor)

# 創建 DataLoader
trn_unnorm = DataLoader(TensorDataset(
    train_x_unnorm, train_y_unnorm), batch_size=1000)
val_unnorm = DataLoader(TensorDataset(
    val_x_unnorm, val_y_unnorm), batch_size=1000)
