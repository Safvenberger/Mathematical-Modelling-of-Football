# Mathematical Modelling of Football

Assignments done in the [Mathematical Modelling of Football course](https://www.uu.se/en/admissions/freestanding-courses/course-syllabus/?kpid=44142&lasar=21%2F22&typ=1) course the fall semester of 2021.

## Assignment 1: Plotting actions
The focal point of the first assignment revolves around plotting actions performed by a specific player during either the 2018 Mens' World Cup or the 2019 Womens' World Cup. The exercise description is as follows:

1. Think of a player who you enjoyed watching at the recent Mens' (2018) or Womens' (2019) World Cups.
2. What actions did they perform that were important and why?
3. Plot the actions and describe how the data supports or contradicts your own analysis.
4. Write a short text using at most two figures that illustrate your point.

**Data**: The data used for this assignment is the [StatsBomb dataset](https://github.com/statsbomb/open-data) for the 2018 Mens' World Cup. 

**Method**: The chosen player for this assignment was Lionel Messi and the actions considered were his involvement in the possession leading to Argentina scoring a goal, as well as Messi's involvement in the shots by Argentina. For the plots and subsequent analysis, the data was limited to their match against France in the Round of 16 (which Argentina lost 4-3).

**Results**: Based on the graphs, his involvement in the Argentinan offense was observed and his contribution to the game, within the actions specified above, could also be examined. 

**Packages used**: Pandas, Matplotlib, MplSoccer

## Assignment 2: Evaluating players
For the second assignment, we were tasked with scouting and evaluating players across the Big 5 European Leagues (Premier League, Ligue 1, Bundesliga, Seria A, La Liga) according to some metric of our own choice, which will then serve as the basis for ranking the players within leagues and ultimately recommending a signing. This assignment was set hypothetically during the summer of 2018 and the instructions were as follows:

1. Implement one of the methods (plus/minus, percentiles, Markov chain, possession chains or one of your own) on the data. Decide on a suitable metric for ranking players and make a top-10 list of players in your position for the whole league.  Write a simple non-technical text (half a page) explaining your metric to the scout and what assumptions it makes. 
2. Use your metric to find a single player in another league (not the Premier league), who you would recommend signing. Add additional statistics and visualisations to explain the strengths and weaknesses of that player (these can use World Cup data where appropriate). Create a two page report, in a poster style, with as many visualisations as you want but max 2 pages on that player.
4. You will be asked to share and present your player for 2 minutes within your group. After the presentations, you will have a group discussion comparing your choices. You should write a single page contrasting your own and the other reports and make a final recommendation on this basis.

**Data**: The data used for this assignment is the [Wyscout dataset](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) for the 2017-2018 season in each respective league.

**Background**: All students were given a specific position and a specific top-flight Premier League team to scout for, and the team I was given was Manchester City and the position to scout for was goalkeeper. Based on the fact that Manchester City already have a high-quality goalkeeper in Ederson during this point in time, the decision was made to instead look for a younger player with potential (and a hopefully smaller price-tag) who can play second-fiddle to Ederson during the important matches, and play rotational minutes in the domestic cup competitions.

**Method**: The metric of choice was Post-Shot Expected Goals minus Goals Allowed (PSxG-GA). This metric is a way to quantify a goalkeepers ability to prevent goals, and for this metric a high positive value is desireable since it implies a goalkeeper concedes fewer goals than expected. Conversely, a negative value means that the goalkeeper concedes more goals than expected. The PSxG model is built upon the expected goals model, where the latter can also be refered to as pre-shot expected goal since it only usese information available before the shot was taken, whereas PSxG also takes into account information available after the shot, such as shot speed and shot trajectory. Worth noting is that for the PSxG model, only shots on target are considered since shots off target will have a PSxG value of 0.

Based on the data and what information was openly available, it was decided to include the following features: 
- Distance to goal,
- Shot angle,
- If the shot was taken with their preferred foot,
- Estimated position of where the shot will hit the target, 
- Estimated time between shot and save attempt.

The choice of model was logistic regression since it, despite it's simplicity, is a common choice for expected goals modelling and typically has high performance. The training of the logistic regression was based on the Premier League data, which thus served as the training data, and the other leagues were then used to validate the model results. Since different type of shots are not equally likely to be goals, separate models were trained for shots taken with feet, headers and free kicks. For penalties, a constant value of 0.76 was used, since this is the [value](https://dataglossary.wyscout.com/xg/) Wyscout use for their xG models. 

**Results**: 

**Possible improvements**: As some of the features used in the model are not as informative as they ideally could be (using regions of the goal rather than pitch and height coordinates, time between shot and save instead of trajectory etc.) the model might have some inherent bias which impacts the results. Another area of improvement could be to consider shots from possesions, rather than treating shots as independent events, since this could deflate or inflate some goalkeepers PSxG if they for instance give many rebounds or save penalties which end up at the feet of the attacking team again. It could also be worth exploring and comparing different models, such as the extreme gradient boosting approach used by [Statsbomb](https://statsbomb.com/2018/11/a-new-way-to-measure-keepers-shot-stopping-post-shot-expected-goals/).

**Packages used**: Pandas, Numpy, Sci-kit learn, Matplotlib, MplSoccer
