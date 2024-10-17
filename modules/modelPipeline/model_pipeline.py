class pipeline():
    #Unprocessed data
    dataset = #hent data

    #Processed data
    data = dataPreprocessor(dataset, config.dataOptions)

    #Untrained model
    untrainedModel = model_selection("Modelname ENUM", config.modelOptions("Modelname ENUM"))

    #Trained model
    predictor = train_model("Modelname ENUM", untrainedModel,data, trainingOptions)

    predictor.predict
