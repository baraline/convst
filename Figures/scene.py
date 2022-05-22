from manim import *
from convst.utils.dataset_utils import load_sktime_dataset
from convst.utils.shapelets_utils import generate_strides_1D, generate_strides_2D
import numpy as np
"""

VERY UGLY CODE, WILL MAKE SOMETHING CLEANER WHEN I'M MORE USED TO THE LIBRARY
(AND WHEN I HAVE SOME TIME)

- Could be nice to be able to visualize with this top shapelets for any dataset
- Compile a video and upload it to RDST github as an introduction
"""


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

def Tex_Group(
        tex_list, color=WHITE, orientation=DOWN, center=False, aligned_edge=LEFT
    ):
    vg = VGroup()
    t_prec = None
    for i in range(len(tex_list)):
        t = Tex(tex_list[i], color=color)
        vg.add(t)
    return vg.arrange(orientation, center=center, aligned_edge=aligned_edge)  
        

def get_input_data(name='GunPoint'):
    return load_sktime_dataset(name)

def shapelet_dist_vect(shp, x, normalize=False, d=1):
    x_strides = generate_strides_1D(x, shp.shape[0], d)
    if normalize:
        shp = (shp - shp.mean()) / shp.std()
        x_strides = (x_strides - x_strides.mean(axis=1, keepdims=True)) / x_strides.std(axis=1, keepdims=True)
    return np.abs(x_strides - shp).sum(axis=1)

class AllScene(Scene):
    def construct(self):
        Slide1().construct()
        Slide2().construct()
        Slide3().construct()
        Slide4().construct()


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
        x_def0 = Tex(r'$X_1 = \{-0.65, \ldots, -0.64\}$, $y_1 = 0$', color=RED).scale(0.65)
        x_def1 = Tex(r'$X_2 = \{-0.78, \ldots, -0.70\}$, $y_2 = 1$', color=BLUE).scale(0.65)
        graph1 = graph_time_series(ax, x1, add_vertex_dots=False, line_color=BLUE)
        x_def1.next_to(graph1, RIGHT, buff=-3).shift(0.5*LEFT)
        x_def1.shift(UP)
        x_def0.next_to(x_def1, UP, buff=0.5)
        graph0 = graph_time_series(ax, x0, add_vertex_dots=False, line_color=RED)
        title = Title("What is time series classification ?")
        tsc_context = VGroup(ax, x_label, x_def0, x_def1, graph1, graph0)

        Text_TSC = Tex_Group(
            [r'Given a set of time series ${\cal X} = \{X_1, \ldots, X_n\}$',
             r'and their classes $Y = \{y_1, \ldots, y_n\}$, how do we build',
             r'a model able to predict the class of new time series ?']
        ).scale(0.525)
        self.wait(0.5)
        self.play(FadeIn(title))
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
        
        graph1 = graph_time_series(ax, x1, add_vertex_dots=False, line_color=BLUE)
        graph0 = graph_time_series(ax, x0, add_vertex_dots=False, line_color=RED)
        DataGrp = VGroup(ax, graph1, graph0)
        title = Title("What are Shapelets ?")
        self.wait(0.5)
        self.add(title)
        self.wait()
        self.play(Create(ax), Create(graph0), Create(graph1), run_time=2)
        self.wait(0.5)
        self.play(DataGrp.animate.stretch(2, 0).shift(UP).scale(0.5))
        self.wait(0.5)
        
        graph_shp = graph_time_series(ax, np.arange(30,55) ,y=x1[30:55], add_vertex_dots=False, line_color=PURPLE, stroke_width=10)
        self.play(Create(graph_shp),run_time=2)
        
        self.play(graph_shp.animate.shift(2*DOWN))
        self.play(graph_shp.animate.shift(2*LEFT))
        
        Text_TSC = Tex_Group(
            [r'A Shapelet is a small time series (i.e. a subsequence), which is often',
             r'extracted from the input time series. We will denote a shapelet as',
             r' $S = \{s_1, \ldots, s_l\}$, with $l$ the length parameter of the shapelet.']
        ).scale(0.575)
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
        
        graph1 = graph_time_series(ax, x1, add_vertex_dots=False, line_color=BLUE)
        graph0 = graph_time_series(ax, x0, add_vertex_dots=False, line_color=RED)
        
        title = Title("How are Shapelets used ?")
        
        graph_shp = graph_time_series(
            ax, np.arange(30,55) ,y=x1[30:55], add_vertex_dots=False, 
            line_color=PURPLE, stroke_width=10
        ).shift(2*DOWN).shift(2*LEFT)
        
        Text_TSC = Tex_Group(
            [r'A Shapelet is a small time series (i.e. a subsequence), which is often',
             r'extracted from the input time series. We will denote a shapelet as',
             r' $S = \{s_1, \ldots, s_l\}$, with $l$ the length parameter of the shapelet.']
        ).scale(0.575)
        Text_TSC.next_to(graph_shp, RIGHT, buff=0.5)
        
        self.add(title, ax, graph1, graph0, graph_shp, Text_TSC)
        self.wait(0.25)
        self.play(FadeOut(Text_TSC), FadeOut(graph_shp), FadeOut(graph1), run_time=2)
        
        shp = x1[30:55]
        shp2 = x0[100:125]
        X_mins = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            X_mins[i,0] = shapelet_dist_vect(shp, X[i,0]).min()
            X_mins[i,1] = shapelet_dist_vect(shp2, X[i,0]).min()
        
        shp_dist0 = shapelet_dist_vect(shp, x0)
        shp_dist1 = shapelet_dist_vect(shp, x1)
        shp_dist2_0 = shapelet_dist_vect(shp2, x0)
        shp_dist2_1 = shapelet_dist_vect(shp2, x1)
        
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
            ax, x0, line_color=RED, add_vertex_dots=False
        )
        
        t = ValueTracker(0)
        
        # Tracker functions
        
        def update_shp():
            i = int(t.get_value())
            graph_shp = graph_time_series(
                ax, np.arange(i,i+shp.shape[0]), y=shp,
                line_color=PURPLE,
                add_vertex_dots=False
            )
            return graph_shp
        
        def get_shp_line_dots():
            step = 1
            i = int(t.get_value())
            vg = VGroup()
            for st in range(0,shp.shape[0],step):
                yp0 = min(shp[st],x0[i+st])
                yp1 = max(shp[st],x0[i+st])
                p1 = Dot(ax.coords_to_point(i+st, yp0))
                p2 = Dot(ax.coords_to_point(i+st, yp1))
                vg.add(DashedLine(start=p1.get_center(), end=p2.get_center(), stroke_color=YELLOW))
            return vg

        def get_dist_vect():
            i = int(t.get_value())
            return graph_time_series(ax_shp0, shp_dist0[:i+1], line_color=RED, add_vertex_dots=False)
                

        l_dots = always_redraw(get_shp_line_dots)
        graph_shp = always_redraw(update_shp)
        x_space = np.arange(shp_dist0.shape[0])
        

        d_vect = always_redraw(get_dist_vect)
        
        
        d_formula = Tex(r'$d(X,S) = \{v_1, \ldots, v_{m-l}\}$,\\ with $v_i = \sum_j^l |x_j - s_j|$').scale(0.6)
        d_formula.next_to(graph_shp, 1.9*DOWN).shift(1.35*RIGHT)
        self.wait(0.25)
        
        self.play(FadeIn(graph_shp))
        
        self.play(Create(l_dots), Create(d_vect))
        
        self.play(FadeIn(d_formula))
        self.wait(6)
        
        self.wait(0.25)
        
        self.play(t.animate.set_value(x_space[-1]), run_time=10, rate_func=rate_functions.ease_in_sine)
        
        self.play(FadeOut(l_dots),FadeOut(graph0),FadeOut(graph_shp),FadeIn(graph1))
        self.wait(1)
        d_vect1 = graph_time_series(ax_shp0, shp_dist1, line_color=BLUE, add_vertex_dots=False)
        self.play(Create(d_vect1))
        
        self.wait()
        min0 = SurroundingRectangle(Dot(ax_shp0.coords_to_point(shp_dist0.argmin(), shp_dist0.min()))).scale(0.65)
        min1 = SurroundingRectangle(Dot(ax_shp0.coords_to_point(shp_dist1.argmin(), shp_dist1.min()))).scale(0.65)
        # link a line to t for dist vect
        
        shp_dist_grp = VGroup()
        shp_dist_grp.add(ax_shp0)
        shp_dist_grp.add(min0)
        shp_dist_grp.add(min1)
        shp_dist_grp.add(d_vect1)
        shp_dist_grp.add(d_vect)
        self.play(Create(min0), Create(min1))
        self.wait()
        
        g_shp = graph_time_series(ax, np.arange(30,55), y=shp, stroke_color=PURPLE, stroke_width=9, add_vertex_dots=False)
        g_shp2 = graph_time_series(ax, np.arange(100,125), y=shp2, stroke_color=PURPLE, stroke_width=9, add_vertex_dots=False)
        min0_tex = Tex(r'$S_1$', color=PURPLE, stroke_width=1.5).scale(0.65)
        min0_tex.next_to(g_shp, DOWN).shift(UP*0.15)
        min1_tex = Tex(r'$S_2$', color=PURPLE, stroke_width=1.5).scale(0.65)
        min1_tex.next_to(g_shp2, UP).shift(DOWN*0.75)
        
        self.play(Create(g_shp), Create(g_shp2))
        self.play(FadeIn(min0_tex), FadeIn(min1_tex))
        self.play(FadeOut(d_formula), shp_dist_grp.animate.stretch(0.5, 0).shift(2.55*LEFT))
        self.wait()
        #Show shp2 dist vect in place of points, repalce by points after
        
        ax_points = Axes(
            x_range=[X_mins[:,0].min(), X_mins[:,0].max(), 2],
            y_range=[X_mins[:,1].min(), X_mins[:,1].max(), 2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).stretch(2, 0).shift(2. * DOWN).scale(0.5).stretch(0.45, 0).shift(2.9*RIGHT)
        self.wait(4)
        self.play(Create(ax_points), run_time=0.5)
        x_label = ax_points.get_x_axis_label(Tex(r"$\min d({\cal X},S_1)$").scale(0.5))
        y_label = ax_points.get_y_axis_label(Tex(r"$\min d({\cal X},S_2)$").scale(0.5))
        
        self.play(FadeIn(x_label),FadeIn(y_label))
        self.wait(4)
        dots1 = Dot(ax_points.c2p(X_mins[2,0],X_mins[2,1]), fill_color=BLUE).scale(0.85)
        dots0 = Dot(ax_points.c2p(X_mins[0,0],X_mins[0,1]), fill_color=RED).scale(0.85)
        self.play(Transform(min0, dots0), Transform(min1, dots1))
        
        all_dots = VGroup()
        for i in range(X.shape[0]):
            if i != 0 and i != 2:
                if y[i] == 0:
                    all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,1]), fill_color=BLUE, fill_opacity=0.75).scale(0.55))
                else:
                    all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,1]), fill_color=RED, fill_opacity=0.75).scale(0.55))
        
        


        self.wait(5)
        self.play(Create(all_dots), run_time=4)
        self.wait()
        
class Slide4(Scene):
    def construct(self):
        txt1 = Tex_Group(
            ['How did this basic model evolve with the SoTA ?',
             '(outside of shapelet generation)']
            ,center=True,aligned_edge=0
        ).scale(0.8)
        self.play(FadeIn(txt1),run_time=1)
        self.wait(4)
        title1 = Title('Extracting more features')
        self.play(FadeOut(txt1), FadeIn(title1),run_time=1)
        txt2 = Tex_Group(
            [r'Given $d(S, X)$, most recent method extract:',
             r'\begin{itemize}\item $\min d(S,X)$ : How close is the best match between $S$ and subsequences of $X$ \item arg$\min d(S,X)$ : Where is the best match located ?\end{itemize}'
             ]
        ).scale(0.7).shift(LEFT*3).shift(2*UP)
        
        X, y, _ = get_input_data()
        x0 = X[0,0] #y1
        x1 = X[2,0] #y0
        
        shp = x1[30:55]
        shp_dist0 = shapelet_dist_vect(shp, x0)
        shp_dist1 = shapelet_dist_vect(shp, x1)
        ax_vect = Axes(
            x_range=[0, shp_dist0.shape[0], 20],
            y_range=[0, shp_dist0.max()+0.1, 10],
            tips=True,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).shift(3.*LEFT).scale(0.5)

        
        g_shp = graph_time_series(ax_vect, shp_dist0, stroke_color=RED, add_vertex_dots=False)
        g_shp1 = graph_time_series(ax_vect, shp_dist1, stroke_color=BLUE, add_vertex_dots=False)
        
        X_mins = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            d =  shapelet_dist_vect(shp, X[i,0])
            X_mins[i,0] = d.argmin()
            X_mins[i,1] = d.min()
        
        ax_points = Axes(
            x_range=[X_mins[:,0].min(), X_mins[:,0].max(), 20],
            y_range=[X_mins[:,1].min(), X_mins[:,1].max(), 2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).shift(3*RIGHT).scale(0.5)
        
        
        dots0 = Dot(ax_points.c2p(X_mins[0,0],X_mins[0,1]), fill_color=RED).scale(0.85)
        min0 = SurroundingRectangle(Dot(ax_vect.coords_to_point(shp_dist0.argmin(), shp_dist0.min()))).scale(0.65)
        dots1 = Dot(ax_points.c2p(X_mins[2,0],X_mins[2,1]), fill_color=BLUE).scale(0.85)
        min1 = SurroundingRectangle(Dot(ax_vect.coords_to_point(shp_dist1.argmin(), shp_dist1.min()))).scale(0.65)
        
        
        all_dots = VGroup()
        for i in range(X.shape[0]):
            if i != 0 and i != 2:
                if y[i] == 0:
                    all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,1]), fill_color=BLUE, fill_opacity=0.75).scale(0.55))
                else:
                    all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,1]), fill_color=RED, fill_opacity=0.75).scale(0.55))
        
        
        self.play(FadeIn(txt2))
        self.wait(3)
        self.play(Create(ax_vect), Create(g_shp), Create(g_shp1))
        self.wait(3)
        self.play(Create(min0),Create(min1))
        self.wait(3)
        self.play(Create(ax_points))
        x_label = ax_points.get_x_axis_label(Tex(r"arg$\min d({\cal X},S_1)$").scale(0.5))
        y_label = ax_points.get_y_axis_label(Tex(r"$\min d({\cal X},S_1)$").scale(0.5))
        
        self.play(FadeIn(x_label),FadeIn(y_label))
        self.play(Transform(min0, dots0),Transform(min1, dots1))
        self.play(Create(all_dots), run_time=4)
        self.wait(4)
        self.play(
            FadeOut(title1),FadeOut(txt2),FadeOut(ax_vect),FadeOut(g_shp),
            FadeOut(ax_points),FadeOut(x_label),FadeOut(dots0),
            FadeOut(y_label),FadeOut(all_dots), FadeOut(min0),
            FadeOut(min1),FadeOut(dots1),FadeOut(g_shp1)
        )
        self.wait()
        
        ###############################################################
        
        title = Title('Using a z-normalized distance (scale invariance)')
        self.play(FadeIn(title),run_time=1)
        
        shp_dist_norm1 = shapelet_dist_vect(shp, x1, normalize=True)
        shp_dist_norm = shapelet_dist_vect(shp, x0, normalize=True)
        shp_normed = (shp - shp.mean()) / shp.std()
        
        ax_input = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[x0.min()-0.1, x0.max()+0.1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).stretch(2, 0).shift(UP).scale(0.5)
        
        ax_dist = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[shp_dist_norm1.min(), shp_dist_norm.max()+2, 5],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(2.58*LEFT)

        ax_shapes = Axes(
            x_range=[0, 20, 2],
            y_range=[-4.5, 4.5, 1],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).shift(3.2*RIGHT).scale(0.5).stretch(0.9,0)
        
        graph_shp =  graph_time_series(ax_shapes, shp_normed, stroke_color=PURPLE, add_vertex_dots=False)
            
        t = ValueTracker(0)
        x_space = np.arange(shp_dist_norm.shape[0])
        def get_dist_vect():
            i = int(t.get_value())
            return graph_time_series(ax_dist, shp_dist_norm[:i+1], stroke_color=RED, add_vertex_dots=False)
                
        def x_subseq_loc():
            i = int(t.get_value())
            x_sub = x0[i:i+shp_normed.shape[0]]
            vg = VGroup()
            vg.add(Dot(ax_input.c2p(i, x_sub[0])))
            vg.add(Dot(ax_input.c2p(i+19, x_sub[19])))
            return SurroundingRectangle(vg, buff=0.01)
        
        def x_subseq():
            i = int(t.get_value())
            x_sub = x0[i:i+shp_normed.shape[0]]
            x_sub_norm = (x_sub - x_sub.mean()) / x_sub.std()
            return graph_time_series(ax_shapes, x_sub_norm, stroke_color=RED, add_vertex_dots=False)
        
        d_vect = always_redraw(get_dist_vect)
        x_subs = always_redraw(x_subseq)
        
        x_subs_loc = always_redraw(x_subseq_loc)
        
        z_norm_txt = Tex_Group(
            [r'$S_{norm} = (S - mean(S)) / std(S)$',
             r'$X_{norm} = (X - mean(X)) / std(X)$']
        ).next_to(x_subs, 0.5*UP).scale(0.5)
        
        g_input = graph_time_series(ax_input, x0, stroke_color=RED, add_vertex_dots=False)
        g_input1 = graph_time_series(ax_input, x1, stroke_color=BLUE, add_vertex_dots=False)
        
        g_shp_dist_norm1 = graph_time_series(ax_dist, shp_dist_norm1, stroke_color=BLUE, add_vertex_dots=False)

        self.play(Create(ax_input), Create(ax_dist),Create(g_input), Create(ax_shapes))
        self.wait()
        self.play(Create(d_vect),Create(graph_shp),Create(x_subs),Create(x_subs_loc))
        self.wait(3)
        self.play(FadeIn(z_norm_txt))
        self.wait(6)
        self.play(FadeOut(z_norm_txt))
        self.play(t.animate.set_value(x_space[-1]), run_time=12, rate_func=rate_functions.ease_in_sine)
        self.wait()
        self.play(FadeOut(x_subs_loc))
        self.wait()
        self.play(Create(g_input1), Create(g_shp_dist_norm1))
        self.wait()
        
        
        ax_points = Axes(
            x_range=[0, X_mins[:,0].max(), 20],
            y_range=[X_mins[:,1].min(), X_mins[:,1].max(), 2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).shift(3*RIGHT).scale(0.5)
        
        X_mins = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            d = shapelet_dist_vect(shp, X[i,0], normalize=True)
            X_mins[i,0] = d.argmin()
            X_mins[i,1] = d.min()
        
        all_dots = VGroup()
        for i in range(X.shape[0]):
            if i != 0 and i != 2:
                if y[i] == 0:
                    all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,1]), fill_color=BLUE, fill_opacity=0.75).scale(0.55))
                else:
                    all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,1]), fill_color=RED, fill_opacity=0.75).scale(0.55))
          
        dots0 = Dot(ax_points.c2p(X_mins[0,0],X_mins[0,1]), fill_color=RED).scale(0.85)
        min0 = SurroundingRectangle(Dot(ax_dist.coords_to_point(shp_dist_norm.argmin(), shp_dist_norm.min()))).scale(0.65)
        dots1 = Dot(ax_points.c2p(X_mins[2,0],X_mins[2,1]), fill_color=BLUE).scale(0.85)
        min1 = SurroundingRectangle(Dot(ax_dist.coords_to_point(shp_dist_norm1.argmin(), shp_dist_norm1.min()))).scale(0.65)
    
        x_label = ax_points.get_x_axis_label(Tex(r"arg$\min d({\cal X},S_1)$").scale(0.5))
        y_label = ax_points.get_y_axis_label(Tex(r"$\min d({\cal X},S_1)$").scale(0.5))
                
        ax_sub_grp = VGroup()
        ax_sub_grp.add(ax_shapes)
        ax_sub_grp.add(x_subs)
        ax_sub_grp.add(graph_shp)
        self.wait()
        self.play(FadeOut(ax_sub_grp),Create(ax_points))
        self.play(FadeIn(x_label),FadeIn(y_label))
        self.play(Create(min0),Create(min1))
        self.wait(3)
        self.play(Transform(min0, dots0), Transform(min1, dots1))
        self.wait()
        self.play(Create(all_dots), run_time=4)
  
        
class Slide5(Scene):
    def construct(self):
        title1 = Title('Our contributions')
        self.play(FadeIn(title1),run_time=1)
        txt2 = Tex_Group(
            [r'\begin{itemize}\item A new feature counting the number of occurences of a shapelet \item Introducing dilated shapelet formulation \item Use both scale sensitive and scale invariant shapelets in the same method \item A fast, open sourced, random shapelet algorithm \end{itemize}']
        ).scale(0.7).shift(UP)
        
        self.play(FadeIn(txt2))
        self.wait(10)
        self.play(FadeOut(txt2), FadeOut(title1))
        
        
        X, y, _ = get_input_data()
        x0 = X[0,0] #y1
        x1 = X[2,0] #y0
        
        shp = x1[30:55]
        threshold = 6
        shp_dist_norm1 = shapelet_dist_vect(shp, x1, normalize=False)
        shp_dist_norm = shapelet_dist_vect(shp, x0, normalize=False)
        
        X_mins = np.zeros((X.shape[0], 3))
        for i in range(X.shape[0]):
            d = shapelet_dist_vect(shp, X[i,0], normalize=False)
            X_mins[i,0] = d.argmin()
            X_mins[i,1] = d.min()
            X_mins[i,2] = (d <= threshold).sum()
        
        ax_input = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[x0.min()-0.1, x0.max()+0.1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).stretch(2, 0).shift(UP).scale(0.5)
        
        ax_dist = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[shp_dist_norm1.min(), shp_dist_norm.max()+2, 5],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(2.58*LEFT)

        ax_points = Axes(
            x_range=[0, X_mins[:,0].max(), 20],
            y_range=[X_mins[:,2].min()-1, X_mins[:,2].max()+1, 15],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).shift(3*RIGHT).scale(0.5)
        c_th = np.zeros(shp_dist_norm1.shape[0])+threshold
        
        
        g_input = graph_time_series(ax_input, x0, stroke_color=RED, add_vertex_dots=False)
        g_input1 = graph_time_series(ax_input, x1, stroke_color=BLUE, add_vertex_dots=False)
        g_shp_dist_norm = graph_time_series(ax_dist, shp_dist_norm, stroke_color=RED, add_vertex_dots=False)
        g_shp_dist_norm1 = graph_time_series(ax_dist, shp_dist_norm1, stroke_color=BLUE, add_vertex_dots=False)
        g_threshold = graph_time_series(ax_dist, c_th, stroke_color=YELLOW, add_vertex_dots=False)
        

        dots0 = Dot(ax_points.c2p(X_mins[0,0],X_mins[0,2]), fill_color=RED).scale(0.85)
        min0 = SurroundingRectangle(Dot(ax_dist.coords_to_point(shp_dist_norm.argmin(), shp_dist_norm.min()))).scale(0.65)
        dots1 = Dot(ax_points.c2p(X_mins[2,0],X_mins[2,2]), fill_color=BLUE).scale(0.85)
        min1 = SurroundingRectangle(Dot(ax_dist.coords_to_point(shp_dist_norm1.argmin(), shp_dist_norm1.min()))).scale(0.65)
    
        all_dots = VGroup()
        for i in range(X.shape[0]):
            if i != 0 and i != 2:
                if y[i] == 0:
                    all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,2]), fill_color=BLUE, fill_opacity=0.75).scale(0.55))
                else:
                    all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,2]), fill_color=RED, fill_opacity=0.75).scale(0.55))
    
        x_area = np.where(shp_dist_norm <= threshold)[0]
        y_area = shp_dist_norm[x_area]
        y_th = c_th[x_area]
        x_area1 = np.where(shp_dist_norm1 <= threshold)[0]
        y_area1 = shp_dist_norm1[x_area1]
        y_th1 = c_th[x_area1]
        
        l1 = []
        l2 = []
        for i in range(x_area.shape[0]):
            l1.append(ax_dist.c2p(x_area[i], y_area[i]))
            l2.append(ax_dist.c2p(x_area[i], y_th[i]))
        l = l1 + l2[::-1]
        area = Polygon(*l, color=GREY, fill_opacity=0.85, stroke_width=0)
        
        l1 = []
        l2 = []
        for i in range(x_area1.shape[0]):
            l1.append(ax_dist.c2p(x_area1[i], y_area1[i]))
            l2.append(ax_dist.c2p(x_area1[i], y_th1[i]))
        l = l1 + l2[::-1]
        area1 = Polygon(*l, color=GREY, fill_opacity=0.85, stroke_width=0)
        
        
        title2 = Title('Shapelet occurence')
        
        
        
        self.play(FadeIn(title2))
        self.play(Create(ax_input),Create(ax_dist),Create(ax_points))
        self.play(Create(g_input))
        self.play(Create(g_shp_dist_norm))
        #Add shapelet display animation
        #Add labels for point ax
        #Add formula for #Occ
        self.play(Create(g_threshold))
        self.wait()
        self.play(FadeIn(area), Create(min0))
        self.wait()
        self.play(Transform(VGroup(area, min0), dots0))
        self.wait()
        self.play(FadeOut(g_input), FadeOut(g_shp_dist_norm), Create(g_input1),Create(g_shp_dist_norm1))
        self.wait()
        self.play(FadeIn(area1), Create(min1))
        self.wait()
        self.play(Transform(VGroup(area1, min1), dots1))
        self.wait()
        self.play(Create(all_dots))
        self.wait()
        
        # How is threshold set 
        
        #Init random
        #Dilation
        #ShpOcc even if not discriminant per say, the number of occurence of a common shape in still a discriminative factor
        #Result time/acc vs SoTa
        #Generalization to Multi/Uneven
        
        
        