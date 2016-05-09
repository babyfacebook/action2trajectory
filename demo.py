import toolbox_qilei.LSTM
#prepare the data
lstm_traj = toolbox_qilei.LSTM.lstm()
lstm_traj.lstm_build()

#initialize the model
lstm_traj.lstm_train()


#train the mdoel