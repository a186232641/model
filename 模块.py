class ACmix(nn.Module):
    def __init__(self, feature_size=64, num_layers=2, dropout=0.1):  # 这个feature_size=200的维度要与输入transformer中的每个单元的维度是一样的
        super(ACmix, self).__init__()
        self.hidden_size = 8  # 初始的向量表示
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))

        self.Linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Linear4 = nn.Linear(self.hidden_size, self.hidden_size)

        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1)
        self.multihead_attn3 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1)
        self.multihead_attn4 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1)

        # self.transformer = nn.Transformer(d_model=64, nhead=1, dropout=0.2)
        self.lstm1 = nn.LSTM(self.hidden_size*2, self.hidden_size*2, num_layers=1)  # RNN
        self.lstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size*2, num_layers=1)  # RNN

        self.linear_1 = nn.Linear(32, 1)
        self.linear_2 = nn.Linear(5, 2)#修改天数
        # self.init_weights()  # nn.Linear 权重参数 初始化
        self.relu = F.relu

    def forward(self, src):
        # print(src.shape)
        # src = torch.transpose(src,0,1)
        # 卷积的输入 [batch_size, in_channels, 4, 4]
        src = src.unsqueeze(1)
        # print(src.shape)

        src_1_k = torch.transpose(self.Linear1(self.conv1(src)).squeeze(1), 0, 1) # torch.Size([32, 1, 5, 7])
        src_1_q = torch.transpose(self.Linear1(self.conv1(src)).squeeze(1), 0, 1) #
        src_1_v = torch.transpose(self.Linear1(self.conv1(src)).squeeze(1), 0, 1)
        # print(src_1_v.shape)  #  torch.Size([5, 32, 7])
        attn_output_conv_1, attn_output_weights_1 = self.multihead_attn1(src_1_q, src_1_k, src_1_v)
        # print(attn_output_conv_1.shape)  #  torch.Size([5, 32, 7])

        src_2_k = torch.transpose(self.Linear2(self.conv2(src)).squeeze(1), 0, 1)
        src_2_q = torch.transpose(self.Linear2(self.conv2(src)).squeeze(1), 0, 1)
        src_2_v = torch.transpose(self.Linear2(self.conv2(src)).squeeze(1), 0, 1)
        # print(src_2_v.shape)  #  torch.Size([5, 32, 7])
        attn_output_conv_2, attn_output_weights_2 = self.multihead_attn2(src_2_q, src_2_k, src_2_v)

        src_3_k = torch.transpose(self.Linear3(self.conv3(src)).squeeze(1), 0, 1)
        src_3_q = torch.transpose(self.Linear3(self.conv3(src)).squeeze(1), 0, 1)
        src_3_v = torch.transpose(self.Linear3(self.conv3(src)).squeeze(1), 0, 1)
        attn_output_Linear_3, attn_output_weights_3 = self.multihead_attn3(src_3_q, src_3_k, src_3_v)
        # print('attn_output_Linear_3',attn_output_Linear_3.shape)

        src_4_k = torch.transpose(self.Linear4(self.conv4(src)).squeeze(1), 0, 1)
        src_4_q = torch.transpose(self.Linear4(self.conv4(src)).squeeze(1), 0, 1)
        src_4_v = torch.transpose(self.Linear4(self.conv4(src)).squeeze(1), 0, 1)

        attn_output_Linear_4, attn_output_weights_4 = self.multihead_attn4(src_4_q, src_4_k, src_4_v)
        # print('attn_output_Linear_4',attn_output_Linear_4.shape)

        # print("attn_output_Linear_2.shape",attn_output_Linear_2.shape)#torch.Size([2, 1, 10, 32])
        # query，key，value的输入形状一定是 [sequence_size, batch_size, emb_size] 比如：value.shape torch.Size( [序列长度,batch_size, 64])
        attn_output_1_2 = torch.cat((attn_output_conv_1, attn_output_conv_2), 2)
        # print("165 attn_output_1_2.shape",attn_output_1_2.shape)
        attn_output_1_2,_=self.lstm1(attn_output_1_2)
        # print('attn_output_1_2.shape',attn_output_1_2.shape)
        attn_output_3_4 = torch.cat((attn_output_Linear_3, attn_output_Linear_4), 2)
        attn_output_3_4,_=self.lstm1(attn_output_3_4)
        # print('attn_output_3_4.shape',attn_output_3_4.shape)

        attn_output_sum=torch.cat((attn_output_1_2, attn_output_3_4), 2)
        attn_output_sum = torch.transpose(attn_output_sum,0,1)
        # print("attn_output_sum.shape",attn_output_sum.shape) # torch.Size([5, 32, 14])
        attn_output = self.linear_2(self.relu(self.linear_1(attn_output_sum).squeeze(2)))  # .squeeze(2)
        # print(attn_output.shape)
        return attn_output
