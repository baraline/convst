from manim import *
from convst.utils.dataset_utils import load_sktime_dataset
from convst.utils.shapelets_utils import generate_strides_1D
import numpy as np


def graph_time_series(ax, x, y=None, **kwargs):
    if y is None:
        return ax.plot_line_graph(
            x_values = np.arange(x.shape[0]),
            y_values = x,
            **kwargs,
        )
    else:
        return ax.plot_line_graph(
            x_values = x,
            y_values = y,
            **kwargs,
        )

def get_input_data(name='GunPoint'):
    return load_sktime_dataset(name)

class Slide1(Scene):
    
    def construct(self):
        X, y, _ = get_input_data()
        x0 = X[0,0]
        x1 = X[2,0]

        ax = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[x0.min()-0.1, x0.max()+0.1, 0.5],
            tips=True,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9)
        
        x_label = ax.get_x_axis_label(Tex("Time").scale(0.65))
        x_def0 = Tex(r'$X_1 = \{-0.65, \ldots, -0.64\}$, $y_1 = 0$', color=BLUE).scale(0.65)
        x_def1 = Tex(r'$X_2 = \{-0.78, \ldots, -0.70\}$, $y_2 = 1$', color=RED).scale(0.65)
        graph1 = graph_time_series(ax, x1, add_vertex_dots=False, line_color=RED)
        x_def1.next_to(graph1, RIGHT, buff=-3)
        x_def1.shift(UP)
        x_def0.next_to(x_def1, UP, buff=0.5)
        graph0 = graph_time_series(ax, x0, add_vertex_dots=False, line_color=BLUE)
        title = Title("What is time series classification ?")
        tsc_context = VGroup(ax, x_label, x_def0, x_def1, graph1, graph0)
        Text_TSC0 = Tex(r'Given a set of time series ${\cal X} = \{X_1, \ldots, X_n\}$')
        Text_TSC1 = Tex('and their classes $Y = \{y_1, \ldots, y_n\}$, how do we build')
        Text_TSC2 = Tex(r'a model able to predict the class of new time series ?')
        Text_TSC1.next_to(Text_TSC0, DOWN, buff=0.5)
        Text_TSC2.next_to(Text_TSC1, DOWN, buff=0.5)
        Text_TSC = VGroup(Text_TSC0, Text_TSC1, Text_TSC2).scale(0.525)
        self.wait(0.5)
        self.add(title)
        self.wait(2)
        self.play(Create(ax), run_time=3)
        self.add(x_label)
        self.wait(2)
        self.play(Create(graph0), run_time=3)
        self.play(FadeIn(x_def0))
        self.wait(2)
        self.play(Create(graph1), run_time=3)
        self.play(FadeIn(x_def1))
        self.wait(2)
        self.play(tsc_context.animate.shift(4 * LEFT).scale(0.5))
        Text_TSC.next_to(tsc_context, RIGHT, buff=0.5)
        self.play(FadeIn(Text_TSC), run_time=2)
        self.wait(8)
    
class Slide2(Scene):
    def construct(self):
        
        X, y, _ = get_input_data()
        x0 = X[0,0]
        x1 = X[2,0]

        ax = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[x0.min()-0.1, x0.max()+0.1, 0.5],
            tips=True,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9)
        
        graph1 = graph_time_series(ax, x1, add_vertex_dots=False, line_color=RED)
        graph0 = graph_time_series(ax, x0, add_vertex_dots=False, line_color=BLUE)
        DataGrp = VGroup(ax, graph1, graph0)
        title = Title("What are Shapelets ?")
        self.wait(0.5)
        self.add(title)
        self.wait()
        self.play(Create(ax), Create(graph0), Create(graph1), run_time=2)
        self.wait(0.5)
        self.play(DataGrp.animate.stretch(2, 0).shift(UP).scale(0.5))
        self.wait(0.5)
        
        graph_shp = graph_time_series(ax, np.arange(32,52) ,y=x1[32:52], add_vertex_dots=False, line_color=PURPLE, stroke_width=10)
        self.play(Create(graph_shp),run_time=2)
        
        self.play(graph_shp.animate.shift(2*DOWN))
        self.play(graph_shp.animate.shift(2*LEFT))
        
        Text_TSC0 = Tex(r'A Shapelet is a small time series (i.e. a subsequence), which is often')
        Text_TSC1 = Tex(r'extracted from the input time series. We will denote a shapelet as')
        Text_TSC2 = Tex(r' $S = \{s_1, \ldots, s_l\}$, with $l$ the length parameter of the shapelet.')
        Text_TSC1.next_to(Text_TSC0, DOWN, buff=0.5)
        Text_TSC2.next_to(Text_TSC1, DOWN, buff=0.5)
        Text_TSC = VGroup(Text_TSC0, Text_TSC1, Text_TSC2).scale(0.575)
        Text_TSC.next_to(graph_shp, RIGHT, buff=0.5)
        self.play(FadeIn(Text_TSC), run_time=2)
        self.wait(8)
        
class Slide3(Scene):
    def construct(self):
        
        X, y, _ = get_input_data()
        x0 = X[0,0]
        x1 = X[2,0]

        ax = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[x0.min()-0.1, x0.max()+0.1, 0.5],
            tips=True,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).stretch(2, 0).shift(UP).scale(0.5)
        
        graph1 = graph_time_series(ax, x1, add_vertex_dots=False, line_color=RED)
        graph0 = graph_time_series(ax, x0, add_vertex_dots=False, line_color=BLUE)
        title = Title("How are Shapelet used ?")
        graph_shp = graph_time_series(ax, np.arange(32,52) ,y=x1[32:52], add_vertex_dots=False, line_color=PURPLE, stroke_width=10).shift(2*DOWN).shift(2*LEFT)
        Text_TSC0 = Tex(r'A Shapelet is a small time series (i.e. a subsequence), which is often')
        Text_TSC1 = Tex(r'extracted from the input time series. We will denote a shapelet as')
        Text_TSC2 = Tex(r' $S = \{s_1, \ldots, s_l\}$, with $l$ the length parameter of the shapelet.')
        Text_TSC1.next_to(Text_TSC0, DOWN, buff=0.5)
        Text_TSC2.next_to(Text_TSC1, DOWN, buff=0.5)
        Text_TSC = VGroup(Text_TSC0, Text_TSC1, Text_TSC2).scale(0.575)
        Text_TSC.next_to(graph_shp, RIGHT, buff=0.5)
        self.add(title, ax, graph1, graph0, graph_shp, Text_TSC)
        self.wait(0.25)
        self.play(FadeOut(Text_TSC), FadeOut(graph_shp), FadeOut(graph1), run_time=2)
        
        shp = x1[32:52]
        shp_dist0 = (generate_strides_1D(x0, shp.shape[0], 1) - shp).sum(axis=1)
        shp_dist1 = (generate_strides_1D(x1, shp.shape[0], 1) - shp).sum(axis=1)
        
        ax_shp0 = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[0, shp_dist0.max()+0.1, 10],
            tips=True,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).stretch(2, 0).shift(2. * DOWN).scale(0.5)

        self.play(Create(ax_shp0), run_time=1)
        graph0 = graph_time_series(
            ax, x0, line_color=BLUE, add_vertex_dots=False
        )
     
        graph_shp = graph_time_series(
            ax, np.arange(0,20), y=shp,
            line_color=PURPLE,
            add_vertex_dots=False
        )
        
        t = ValueTracker(10)
        
        graph_shp.add_updater(lambda x: x.move_to(ax.c2p(t.get_value(), shp[14])))
        x_space = np.arange(shp_dist0.shape[0]+10)
        
        def get_shp_line_dots0():
            i = int(t.get_value()) - 9
            yp0 = min(shp[0],x0[i])
            yp1 = max(shp[0],x0[i])
            p1 = Dot(ax.coords_to_point(i, yp0), radius=0.02)
            p2 = Dot(ax.coords_to_point(i, yp1), radius=0.02)
            return DashedLine(start=p1.get_left(), end=p2.get_left(), stroke_color=YELLOW)
        
        def get_shp_line_dots1():
            i = int(t.get_value()) - 4
            yp0 = min(shp[4],x0[i])
            yp1 = max(shp[4],x0[i])
            p1 = Dot(ax.coords_to_point(i, yp0), radius=0.02)
            p2 = Dot(ax.coords_to_point(i, yp1), radius=0.02)
            return DashedLine(start=p1.get_center(), end=p2.get_center(), stroke_color=YELLOW)
        
        def get_shp_line_dots2():
            i = int(t.get_value()) 
            yp0 = min(shp[9],x0[i])
            yp1 = max(shp[9],x0[i])
            p1 = Dot(ax.coords_to_point(i, yp0), radius=0.02)
            p2 = Dot(ax.coords_to_point(i, yp1), radius=0.02)
            return DashedLine(start=p1.get_center(), end=p2.get_center(), stroke_color=YELLOW)
        
        def get_shp_line_dots3():
            i = int(t.get_value()) + 4
            yp0 = min(shp[14],x0[i])
            yp1 = max(shp[14],x0[i])
            p1 = Dot(ax.coords_to_point(i, yp0), radius=0.02)
            p2 = Dot(ax.coords_to_point(i, yp1), radius=0.02)
            return DashedLine(start=p1.get_center(), end=p2.get_center(), stroke_color=YELLOW)
        
        def get_shp_line_dots4():
            i = int(t.get_value()) + 9
            yp0 = min(shp[19],x0[i])
            yp1 = max(shp[19],x0[i])
            p1 = Dot(ax.coords_to_point(i, yp0), radius=0.02)
            p2 = Dot(ax.coords_to_point(i, yp1), radius=0.02)
            return DashedLine(start=p1.get_right(), end=p2.get_right(), stroke_color=YELLOW)
        
        
        l_dots0 = always_redraw(get_shp_line_dots0)
        l_dots1 = always_redraw(get_shp_line_dots1)
        l_dots2 = always_redraw(get_shp_line_dots2)
        l_dots3 = always_redraw(get_shp_line_dots3)
        l_dots4 = always_redraw(get_shp_line_dots4)
        self.wait(0.25)
        
        self.add(graph_shp)
        #place 
        self.play(Create(l_dots0),Create(l_dots1),Create(l_dots2),Create(l_dots3),Create(l_dots4))
        self.wait(2)
        #add formula fade in out
        self.play(t.animate.set_value(x_space[-1]), run_time=6)
        # link a line to t for dist vect
        self.wait()

        #ani_group = Succession(*[Transform(r[i], r[i+1]) for i in range(0,4)])
        #self.play(ani_goup)
        #Reduce shp width, place at index 0, disapear X1, make points appear on series, dashed line between shp and X, display formula
        #Rolling to produce distance vector for one X
        #disapear X0, same process to obtain distance vector, place the two dist vector on separate axises (scale 1/2 for the first rolling then scale 
        #it to 1/4 and place 2nd axis)
        
        #Slide 3
        #How are Shapelet used ?
        #Sliding window to distance vector
        
        
        #Text to explain what the goal is
        #make a sliding, which produce the distance vector for the two X
        #highligh min to show class difference
        #Make the case on scale sensitive/z-norm
        #Case of localization

        #Init random
        #Dilation
        #ShpOcc even if not discriminant per say, the number of occurence of a common shape in still a discriminative factor
        #Result time/acc vs SoTa
        #Generalization to Multi/Uneven
        
        
        