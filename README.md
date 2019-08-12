# TimeSeries forecasting using Facebook Prophet

Prerequisites: 

1. Python 3.6.x
2. Microsoft Visual C++ Redistributable for Visual Studio 2017 ( https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017  )
3. Build tools for visual studio 2017 ( https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017  )
4. pip install fbprophet 

Use Case:

To forecast the total charged volume by a particular TADIG Code.

Solution:

There are multiple TAGID Codes, hence initially we have grouped by the TADIG code with Process Date to find the sum of "Total Charged Volume(MB)".
Then we have created a Prophet model to do the forecasting for next 2 days.

Holidays can also be added in this model, which isn't used in this particular example.

Note: The data is confidential, hence have used a snippet of it.
But this solution will definitely clear the concept of "How to use FB Prophet"


![Test Image 7](https://github.com/pik1989/TimeSeries/blob/master/Image.png)
