# Machine-Learning-Model-space-rats
This project is an extension of AI-project 2. With the same ruleset of AI project 2, we created a machine learning model that, given a certain game state (with the knowledge bases of the rat probabilities and the bot possible locations), predicts how many actions are left for the bot to reach the rat. 

PREVIOUS RULESET REMINDER: 
We generated a 30x30 grid with opened and closed cells. Each ship was generated so open cells were interconnected (that is, any open cell can be reached from any other open cell). We then placed a bot at a random open cell and a rat in a separate open cell. The goal of project 2 was to make the bot reach the rat. The bot had three actions it could take: scan cells, attempt ping, step. The bot does not have full information of the grid. 

A more complete description of project 2 is in a PDF file labeled "Project 2 Instructions"


**How you are representing your input data:**
Our input data consists of two knowledge bases, the KB of the possible bot locations, and the KB of the rat location probabilities. Each knowledge base is a dxd matrix (in this case d=30) which represents the ship with open cells and closed cells. For the KB with the possible bot locations, a 0 represents a cell that is closed, a 1 represents a cell that is open but the bot cannot possibly be there, and 2 represents a cell in which the bot could be. For the KB with the rat location probabilities, each cell consists of a number between 0 and 1, which represents the probability that the rat is located in that cell. After any action the bot takes (cell sensing, pinging, or walking) we reshape the bot location and rat location probability knowledge bases into 1x900 vectors. We concatonate the flattened bot KB and the rat KB, respectively. This new 1x1800 vector is our input data.

**How you are representing your output data**
Our output data is a scalar value which, given a certain game state, represents the bot's prediction of how many moves are left to reach the rat from the initial state.


**The model architecture you are using** 
The structure we chose for the neural network were a sequence of linear layers. We started out with an input layer of 1800 features, this would take in the 1800 data points that we collected for our input. Then a layer passes those 1800 features to 900 features. Then three more layers from 900 to 400, 400 to 50, and finally an output layer that takes the 50 features and boils them down to one prediction feature. The reason why we decided on this gradual descent of features is because we didn't want too much information to be lost when passed from one layer to another. We felt that this was a sufficient number of layers to get an accurate prediction without costing us too much in terms of training time. We also put a sigmoid function between each hidden layer to introduce some non-linearity between the layers. This is to prevent the network from searching for linear patterns in the data, allowing the network to develop more complex patterns. The reason we landed on sigmoid functions in particular is that most of our data is going to be between 0 and 2, and we found that the sigmoid function has interesting values at these points.

**The loss you are using to evaluate your model**
We utilized the Mean Squared Error loss function to calculate the loss. We chose this loss function because we want the network to make a numerical prediction as opposed to a classification. we are interested in how close the network's estimate can get to the actual value. The closer the network can get to the actual number of moves, the lower the mse loss will be.


**How you are collecting your data**
We collected our data in one very long vector of concatonated tensors. We had the bot play out different games with randomized rat and bot locations. After every action of the bot, we would take a snapshot of the rat and bot location knowledge bases and add them to the tensor of data. These knowledge bases would be the x data that we will feed into the neural network. We would also keep track of the number of actions the bot has taken over the game. After the game is over and we know how many actions the bot used, we can subtract the number of moves the bot made at each given time from the total number of moves to get the number of moves left. This would constitute our y data.
