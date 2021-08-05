# README of English Version

Project code repository for few-shot malware classification based on fused analysis features. This repository contains all the detailed implementations including dataset preprocessing, encapsulation for task-level meta-learning scheme, proposed models and baselines, training/testing running codes. The features of this project are:

- **High-Level Encapsulation for Models as Loose-Coupled Layered Implementation**: Some fundamental model-agnostic abilities can be obtained by inheriting corresponding father models, such as sequence and image embedding, parsing of task parameters. This enables fast development of new models.
- **High-Level Encapsulation for Meta-Learning Task Episode Sampling**: Meta-learning task sampling is encapsulated into task instance which allows defining costumed sampling rule and training/testing episode can be conveniently sampled by calling an interface.
- **Fully Parameter-Configurable Model Training and Testing Support**: All the training/testing related parameters are all read from readable JSON configuration file, such as hidden layer dimension, training epoch even embedding backbone. The behaviors can be completely controlled by editing JSON configuration file and this ensures replicability.
- **Modular Model Structure Parsing and Building**: Using high-level builder interface to make model component instances such as model object, optimizer and feature fusion module. This isolates modifications of these components from running code and makes it convenient to add new components. 
- **Detailed Running and Testing Log Recoding**: Easily manage a running training/testing instance as 'dataset + version' form. Logs are traced and dumped every time launching training/testing and this makes statistics of results very convenient.
- **Abundant Configurable Parameter Support**: A lot number of parameters can be configured in JSON config file, even as Visdom visualization, console printing of results, gradient clipping and GPU device setting and etc.
- **Support fot Automatic Task Running**: Supports for using preset configured task to automatically running some training/testing tasks offline. This allows appointment a lot of tasks to run and server can automatically run them without human operation even in night.

---

(under construction...)