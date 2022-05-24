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

########################################
#                                      #
#              CONSTANTS               #
#                                      # 
########################################

COLOR_0 = RED
COLOR_1 = BLUE
COLOR_SHP = PURPLE

########################################
#                                      #
#        DATA HELPER FUNCTIONS         #
#                                      # 
########################################


def shapelet_dist_vect(shp, x, normalize=False, d=1):
    x_strides = generate_strides_1D(x, shp.shape[0], d)
    if normalize:
        shp = (shp - shp.mean()) / shp.std()
        x_strides = (x_strides - x_strides.mean(axis=1, keepdims=True)) / x_strides.std(axis=1, keepdims=True)
    return np.abs(x_strides - shp).sum(axis=1)


def get_input_data(name='GunPoint'):
    return load_sktime_dataset(name)
        

def range_from_x(x, n_steps=10, min_to=None, max_to=None):
    if min_to is None:
        v_min = x.min()
    else:
        v_min = min_to
        
    if max_to is None:
        v_max = x.max()
    else:
        v_max = max_to    
    
    v_range = v_max-v_min
    step = v_range/n_steps
    
    return [v_min, v_max, step]

def get_problem_data(name='GunPoint', default_shp_len=0.1, index_0=None, index_1=None):
    X, y, _ = get_input_data(name=name)
    x0 = X[np.random.choice(np.where(y==0)[0])]
    x1 = X[np.random.choice(np.where(y==1)[0])]
    
    if index_0 is None:
        l0 = int(x0.shape[0] * default_shp_len)
        i_shp_0 = np.random.choice(x0.shape[0] - l0)
        index_0 = np.arange(i_shp_0,i_shp_0+l0)
    shp0 = x0[index_0]
    
    if index_1 is None:
        l1 = int(x1.shape[0] * default_shp_len)
        i_shp_1 = np.random.choice(x0.shape[0] - l1)
        index_1 = np.arange(i_shp_1,i_shp_1+l1)
    shp1 = x1[index_1]
    
    return X, y, x0, x1, shp0, shp1

def get_features(X, shp, normalize=False, d=1, threshold=10):
    X_mins = np.zeros((X.shape[0], 3))
    for i in range(X.shape[0]):
        d_vect = shapelet_dist_vect(shp, X[i,0], normalize=normalize, d=d)
        X_mins[i,0] = shapelet_dist_vect(shp, X[i,0]).min()
        X_mins[i,1] = shapelet_dist_vect(shp2, X[i,0]).argmin()
        X_mins[i,2] = (shapelet_dist_vect(shp2, X[i,0])<threshold).sum()
    return X_mins

########################################
#                                      #
#       GRAPHIC HELPER FUNCTIONS       #
#                                      # 
########################################


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

    
def get_axis_from_x(
        x, x_label, y_label, y=None, min_to=None, max_to=None ,tips=False,
        color=GREEN, numbered=True, label_color=WHITE, **kwargs
    ):
    x_range = range_from_x(np.arange(x.shape[0]))
    if y is None:
        y_range = range_from_x(x)
    else:
        y_range = range_from_x(y)
        
    ax = Axes(
        x_range=x_range,
        y_range=y_range,
        tips=tips,
        axis_config={
            "include_numbers": numbered,
            "color":color,
            **kwargs
        },
    ).scale(0.9)
    
    
    labels = graph.get_axis_labels(
        x_label_tex=x_label, y_label_tex=y_label
    ).set_color(label_color).scale(0.6)
    
    return VGroup(ax,labels)
    
def get_211_axes(
        x_ranges, y_ranges, x_labels, y_labels,
        tips=False, color=GREEN, numbered=True, label_color=WHITE,
        **kwargs
    ):
    
    ax0 = Axes(
        x_range=x_ranges[0],
        y_range=y_ranges[0],
        tips=tips,
        axis_config={
            "include_numbers": numbered,
            "color":color,
            **kwargs
        },
    ).scale(0.9).stretch(2, 0).shift(UP).scale(0.5)
    labels0 = graph.get_axis_labels(
        x_label_tex=x_labels[0], y_label_tex=y_labels[0]
    ).set_color(label_color).scale(0.6)
    
    
    ax1 = Axes(
        x_range=x_ranges[1],
        y_range=y_ranges[1],
        tips=tips,
        axis_config={
            "include_numbers": numbered,
            "color":color     ,
            **kwargs
        },
    ).scale(0.9).shift(2*DOWN).shift(2.75*LEFT).scale(0.5)
    labels1 = graph.get_axis_labels(
        x_label_tex=x_labels[1], y_label_tex=y_labels[1]
    ).set_color(label_color).scale(0.6)


    ax2 = Axes(
        x_range=x_ranges[2],
        y_range=y_ranges[2],
        tips=tips,
        axis_config={
            "include_numbers": numbered,
            "color":color,
            **kwargs
        },
    ).scale(0.9).shift(2*DOWN).shift(2.75*RIGHT).scale(0.5)
    labels2 = graph.get_axis_labels(
        x_label_tex=x_labels[2], y_label_tex=y_labels[2]
    ).set_color(label_color).scale(0.6)
    
    return VGroup(ax0,labels0), VGroup(ax1,labels1), VGroup(ax2,labels2)
    
def get_numerotation(i):
    #Calibrate so it is leftmost to the title at the bar level
    return Tex('{}'.format(i)).scale(0.6).shift(2*UP).shift(4*LEFT)

def make_title(title, i):
    return VGroup(Title(title), get_numerotation(i))
    

class Slide(Scene):
    def construct(self):
        X, y, x0, x1, shp0, shp1 = get_problem_data(
            index_0=np.arange(30,55), index_1=np.arange(100,125)
        )
        shp0_norm = (shp0 - shp0.mean())/shp0.std()
        shp1_norm = (shp1 - shp1.mean())/shp1.std()
        
        dist_vect_shp_0_0 = shapelet_dist_vect(shp0, x0, normalize=False, d=1)
        dist_vect_shp_0_1 = shapelet_dist_vect(shp0, x1, normalize=False, d=1)
        norm_dist_vect_shp_0_0 = shapelet_dist_vect(shp0, x0, normalize=False, d=1)
        norm_dist_vect_shp_0_1 = shapelet_dist_vect(shp0, x1, normalize=False, d=1)
        
        dist_vect_shp_1_0 = shapelet_dist_vect(shp1, x0, normalize=False, d=1)
        dist_vect_shp_1_1 = shapelet_dist_vect(shp1, x1, normalize=False, d=1)
        norm_dist_vect_shp_1_0= shapelet_dist_vect(shp1, x0, normalize=False, d=1)
        norm_dist_vect_shp_1_1 = shapelet_dist_vect(shp1, x1, normalize=False, d=1)
        
        X_T_0 = get_features(X, shp0, normalize=False, d=1, threshold=10)
        X_T_1 = get_features(X, shp1, normalize=False, d=1, threshold=10)

        ########################################
        #                                      #
        #          INTRODUCE SHAPELETS         #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("What are Shapelets ?", 1)
        vg_all.add(*title)
        
        ax = get_axis_from_x(x0, Tex('Time'), Tex('Value'), kwargs)

        graph_x0 = graph_time_series(ax, x0, add_vertex_dots=False, line_color=COLOR_0)
        graph_x1 = graph_time_series(ax, x1, add_vertex_dots=False, line_color=COLOR_1)
        graph_shp0 = graph_time_series(
            ax, np.arange(30,55) ,y=shp0, add_vertex_dots=False,
            line_color=COLOR_SHP, stroke_width=10
        )

        DataGrp = VGroup(ax, graph_x0, graph_x1)
        vg_all.add(*DataGrp)
        self.FadeIn(title)
        self.wait()
        self.play(Create(ax), run_time=2)
        self.play(Create(graph0), run_time=2)
        self.play(Create(graph1), run_time=2)
        self.wait(0.5)
        self.play(DataGrp.animate.stretch(2, 0).shift(UP).scale(0.5))
        self.wait(0.5)
        
        self.play(Create(graph_shp0),run_time=2)
        self.play(graph_shp.animate.shift(2*DOWN))
        self.play(graph_shp.animate.shift(2*LEFT))
        
        Tex_shp = Tex(
            r' $S = \{s_1, \ldots, s_l\}$'
        ).scale(0.8).next_to(graph_shp_1, RIGHT)
        vg_all.add(*Tex_shp)
        self.play(FadeIn(Tex_shp))
        self.wait(4)
        self.play(FadeOut(vg_all))
        self.remove(vg_all)
        self.wait(1)
        ########################################
        #                                      #
        #       SHOW HOW TO EXTRACT MIN        #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("How are Shapelets ?", 2)
        vg_all.add(*title)
        
        x_ranges = [
            [0,np.arange(x0.shape[0]),20],
            [0,np.arange(dist_vect_shp_1_0.shape[0]),20],
            range_from_x(X_T_0[:,0])
        ]
        
        y_ranges = [
            range_from_x(x0),
            range_from_x(dist_vect_shp_1_0),
            range_from_x(X_T_1[:,0])
        ]
        
        x_labels = [
            Tex(''),
            Tex(''),
            Tex(r"$\min d({\cal X},S_1)$")
        ]
        
        y_labels = [
            Tex(''),
            Tex(''),
            Tex(r"$\min d({\cal X},S_2)$")
        ]
        
        ax0, ax1, ax2 = get_211_axes(x_ranges, y_ranges, x_labels, y_labels)
        vg_all.add(*[ax0, ax1, ax2])
        graph_x0 = graph_time_series(ax0, x0, add_vertex_dots=False, line_color=COLOR_0)
        graph_x1 = graph_time_series(ax0, x1, add_vertex_dots=False, line_color=COLOR_1)
        d_vect_0_0 = graph_time_series(ax1, dist_vect_shp_0_0, line_color=COLOR_0, add_vertex_dots=False)
        graph_shp1 = graph_time_series(ax0, np.arange(100,125), y=shp1, line_color=COLOR_1, add_vertex_dots=False)
        t = ValueTracker(0)
        x_space = np.arange(dist_vect_shp_1_0.shape[0])
        
        def update_shp():
            i = int(t.get_value())
            graph_shp = graph_time_series(
                ax, np.arange(i,i+shp.shape[0]), y=shp,
                line_color=COLOR_SHP,
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
            return graph_time_series(ax1, dist_vect_shp_1_0[:i+1], line_color=COLOR_1, add_vertex_dots=False)
                

        l_dots = always_redraw(get_shp_line_dots)
        graph_shp = always_redraw(update_shp)
        d_vect = always_redraw(get_dist_vect)
        
        d_formula = Tex_Group(
            [r'$d(X,S) = \{v_1, \ldots, v_{m-(l-1)}\}$',
             r'with $v_i = \sum_{j=1}^l |x_{i+(j-1)} - s_j|$']
        ).scale(0.6).next_to(ax1,RIGHT).shift(RIGHT)
        
        min0 = SurroundingRectangle(Dot(ax2.coords_to_point(dist_vect_shp_0_0.argmin(), dist_vect_shp_0_0.min()))).scale(0.65)
        min1 = SurroundingRectangle(Dot(ax2.coords_to_point(dist_vect_shp_0_1.argmin(), dist_vect_shp_0_1.min()))).scale(0.65)
        
        self.play(Create(ax0), Create(graph_x0))
        self.play(Create(ax1))
        self.play(Create(graph_shp), Create(l_dots), Create(d_vect))
        self.wait(1)
        self.play(FadeIn(d_formula))
        self.wait(4)
        self.play(t.animate.set_value(x_space[-1]), run_time=10, rate_func=rate_functions.ease_in_sine)
        self.wait(1)
        self.play(Create(graph_x1))
        self.play(Create(d_vect_0_0))
        self.wait(1)
        self.play(Create(min0, min1))
        self.wait(1)
        
        self.play(Create())
        self.play(FadeOut(d_formula), Create(ax2))
        self.play(Create())
        
        
"""   
        
class Slide3(Scene):
    def construct(self):
    
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
            x_range=[0, 25, 2],
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
        shp_dist_norm1 = shapelet_dist_vect(shp, x1, normalize=True)
        shp_dist_norm = shapelet_dist_vect(shp, x0, normalize=True)
        
        X_mins = np.zeros((X.shape[0], 3))
        for i in range(X.shape[0]):
            d = shapelet_dist_vect(shp, X[i,0], normalize=True)
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
            y_range=[0, X_mins[:,2].max()+1, 5],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(3*RIGHT)
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
        x_label = ax_points.get_x_axis_label(Tex(r"arg$\min d({\cal X},S_1)$").scale(0.5))
        y_label = ax_points.get_y_axis_label(Tex(r"$\# d({\cal X},S_1)<\lambda$").scale(0.5))
        
        self.play(FadeIn(title2))
        self.play(Create(ax_input),Create(ax_dist),Create(ax_points))
        self.play(FadeIn(x_label),FadeIn(y_label))
        self.play(Create(g_input))
        self.play(Create(g_shp_dist_norm))
        self.play(Create(g_threshold))
        self.wait()
        self.play(FadeIn(area), Create(min0))
        self.wait()
        vare = VGroup(area, min0)
        self.play(Transform(vare, dots0))
        self.wait()
        self.play(FadeOut(g_input), FadeOut(g_shp_dist_norm), Create(g_input1),Create(g_shp_dist_norm1))
        self.wait()
        self.play(FadeIn(area1), Create(min1))
        self.wait()
        vare1 = VGroup(area1, min1)
        self.play(Transform(vare1, dots1))
        self.wait()
        self.play(Create(all_dots))
        self.wait()
        self.play(
            FadeOut(title2),FadeOut(all_dots), FadeOut(vare), FadeOut(vare1),
            FadeOut(dots1), FadeOut(dots0), FadeOut(min0), FadeOut(min1),
            FadeOut(g_threshold), FadeOut(x_label), FadeOut(g_shp_dist_norm1), 
            FadeOut(g_input1), FadeOut(ax_input),
            FadeOut(ax_points), FadeOut(y_label), FadeOut(ax_dist)
        )
        self.wait()
        title2 = Title('Dilated Shapelet')
        self.play(FadeIn(title2))
        
        shp = x1[35:40]
        
        ax_dist = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[0, 20, 5],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(2.58*LEFT)

        ax_points = Axes(
            x_range=[0, x0.shape[0], 20],
            y_range=[0, 6, 1],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(3*RIGHT)
        
        self.play(Create(ax_input), Create(ax_dist), Create(g_input), Create(ax_points))
        d_formula = Tex(r'$d(X,S) = \{v_1, \ldots, v_{m-(l-1)\times d}\}$,\\ with $v_i = \sum_{j=1}^l |x_{i+(j-1)\times d} - s_j|$').scale(0.55)
        d_formula.next_to(g_input1, UP).shift(3.05*LEFT).shift(DOWN)
        self.play(FadeIn(d_formula))
        for d in [1,5,11]:
            shp_dist_norm0 = shapelet_dist_vect(shp, x0, normalize=False, d=d)
            t = ValueTracker(0)
            
            # Tracker functions
            
            def update_shp():
                i = int(t.get_value())
                vg = VGroup()
                for j in range(shp.shape[0]):
                    vg.add(Dot(ax_input.c2p(i+(j*d), shp[j]),fill_color=YELLOW).scale(0.8))
                return vg
            
            def get_shp_line_dots():
                step = 1
                i = int(t.get_value())
                vg = VGroup()
                for st in range(0,shp.shape[0],step):
                    yp0 = min(shp[st],x0[i+(d*st)])
                    yp1 = max(shp[st],x0[i+(d*st)])
                    p1 = Dot(ax_input.coords_to_point(i+(d*st), yp0))
                    p2 = Dot(ax_input.coords_to_point(i+(d*st), yp1))
                    vg.add(DashedLine(start=p1.get_center(), end=p2.get_center(), stroke_color=YELLOW))
                return vg
    
            def get_dist_vect():
                i = int(t.get_value())
                return graph_time_series(ax_dist, shp_dist_norm0[:i+1], line_color=RED, add_vertex_dots=False)
                    
            
            def brace_d():
                i = int(t.get_value())
                vg = VGroup(
                    Dot(ax_input.c2p(i+0.5, shp[0]+0.15)),
                    Dot(ax_input.c2p(i+d-0.5, shp[1]+0.15))
                )
                brace = Brace(vg)
                label = Text('d = {}'.format(d)).scale(0.35)
                label.next_to(brace,DOWN, buff=0.01)
                return VGroup(brace,label)
            
            
            brace = always_redraw(brace_d)
            l_dots = always_redraw(get_shp_line_dots)
            graph_shp = always_redraw(update_shp)
            x_space = np.arange(shp_dist_norm0.shape[0])
            d_vect = always_redraw(get_dist_vect)
            
            self.play(Create(l_dots),Create(graph_shp),Create(d_vect), Create(brace))
            self.play(t.animate.set_value(x_space[-1]), run_time=6, rate_func=rate_functions.ease_in_sine)
            
            X_mins = np.zeros((X.shape[0], 2))
            for i in range(X.shape[0]):
                v = shapelet_dist_vect(shp, X[i,0], normalize=False, d=d)
                X_mins[i,0] = v.argmin()
                X_mins[i,1] = v.min()
            
            
            
            
            all_dots = VGroup()
            for i in range(X.shape[0]):
                if i != 0 and i != 2:
                    if y[i] == 0:
                        all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,1]), fill_color=BLUE, fill_opacity=0.75).scale(0.55))
                    else:
                        all_dots.add(Dot(ax_points.c2p(X_mins[i,0],X_mins[i,1]), fill_color=RED, fill_opacity=0.75).scale(0.55))
            self.play(FadeIn(all_dots))
            self.wait(2)
            self.play(FadeOut(l_dots),FadeOut(graph_shp),FadeOut(d_vect), FadeOut(all_dots), FadeOut(brace))
        
            
        
def boxed_txt(txt):
    text = Tex(txt)
    framebox = SurroundingRectangle(text, buff = .1, stroke_color=GREY)
    return VGroup(text, framebox)
    
            
class Slide6(Scene):
    def construct(self):
        title1 = Title('Random Dilated Shapelet Transform')
        self.play(FadeIn(title1),run_time=1)
        
        b7 = boxed_txt('7')
        b9 = boxed_txt('9')
        b11 = boxed_txt('11')
        g_length = VGroup(
            b7,b9,b11
        ).arrange(RIGHT, buff=0.1)
        txt_length = Tex('Length')
        txt_length.next_to(g_length, UP, buff=0.1)
        g_length.add(txt_length)
        
        
        b5 = boxed_txt('5')
        b10 = boxed_txt('10')
        g_thresd = VGroup(
            b5,b10
        ).arrange(RIGHT, buff=0.1)
        txt_th = Tex('$P_0 ,P_1$')
        txt_th.next_to(g_thresd, UP, buff=0.1)
        g_thresd.add(txt_th)
        
        
        b03 = boxed_txt('0.3')
        b07 = boxed_txt('0.7')
        g_pnorm = VGroup(
            b03,b07
        ).arrange(RIGHT, buff=0.1)
        txt_p = Tex('$p_{norm}$')
        txt_p.next_to(g_pnorm, UP, buff=0.1)
        g_pnorm.add(txt_p)
        
        txt_in=Tex('$({\cal X}, Y)$')
        txt_in_T=Tex('$({\cal X}, Y)$')
        txt_n_shp=Tex('$N_{shapelets}$')
        
        self.play(Create(g_length))
        self.wait()
        self.play(g_length.animate.scale(0.8).shift(5*LEFT).shift(1.75*UP))
        self.wait()
        self.play(Create(g_thresd))
        self.wait()
        self.play(g_thresd.animate.scale(0.8).shift(2.5*LEFT).shift(1.75*UP))
        self.wait()
        self.play(Create(g_pnorm))
        self.wait()
        self.play(g_pnorm.animate.scale(0.8).shift(1.75*UP))
        self.wait()
        self.play(Create(txt_in))
        self.wait()
        self.play(txt_in.animate.scale(0.8).shift(2.5*RIGHT).shift(1.75*UP))
        self.add(txt_in_T.scale(0.8).shift(2.5*RIGHT).shift(1.75*UP))
        self.wait()
        self.play(Create(txt_n_shp))
        self.wait()
        self.play(txt_n_shp.animate.scale(0.8).shift(5*RIGHT).shift(1.75*UP))
        self.wait()
        
        X, y, _ = get_input_data()
        x0 = X[0,0] #y1
        x1 = X[2,0] #y0
        
        shp = x1[[100,103,106,109,112,115,118]]
        shp_norm = (shp - shp.mean()) / shp.std()
        ax_shp = Axes(
            x_range=[0, shp.shape[0], 1],
            y_range=[shp_norm.min(), shp_norm.max(), 0.5],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(2.9*LEFT)

        ax_input = Axes(
            x_range=[0, x1.shape[0], 20],
            y_range=[x1.min(), x1.max(), 0.5],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(3.2*RIGHT)
        x_input = graph_time_series(ax_input, x1, line_color=BLUE, add_vertex_dots=False)
        self.play(b7[1].animate.set_stroke_color(GREEN), b7[0].animate.set_fill(GREEN))
        self.play(Transform(b7[1],ax_shp))
        self.wait(2)
        self.play(txt_in_T.animate.set_fill(GREEN))
        self.play(FadeOut(txt_in_T),FadeIn(ax_input), FadeIn(x_input))
        self.wait(2)
        
        txt_dil = Tex(r'd = $\lfloor 2^{x} \rfloor$, $x \in [0, log \frac{||X||}{l}]$').scale(0.8).shift(0.5*UP)
        txt_dil1 = Tex(r'd = 3').scale(0.8).shift(0.5*UP).shift(5*LEFT)
        self.play(Create(txt_dil))
        self.wait(2)
        self.play(Transform(txt_dil, txt_dil1))
        self.wait(2)
        txt_point = Tex(r'$ i \in [0, ||X|| - (l-1)\times d]$').scale(0.8).shift(0.5*UP)
        txt_point3 = Tex(r'$ i = 100 $').scale(0.8).shift(0.5*UP).shift(5*RIGHT)
        self.play(Create(txt_point))
        self.wait(2)
        self.play(Transform(txt_point, txt_point3))
        self.wait(2)
        vPoints = VGroup()
        for ix in [100,103,106,109,112,115,118]:
            vPoints.add(Dot(ax_input.c2p(ix, x1[ix]),fill_color=PURPLE).scale(0.9))
        self.play(Create(vPoints, run_time=5, rate_func=rate_functions.ease_in_sine))
        self.wait()
        g_shp = graph_time_series(ax_shp, shp, line_color=PURPLE, add_vertex_dots=False)
        shp_norm = (shp - shp.mean()) / shp.std()
        g_shp_norm = graph_time_series(ax_shp, shp_norm, line_color=PURPLE, add_vertex_dots=False)
        self.play(b07[1].animate.set_stroke_color(GREEN), b07[0].animate.set_fill(GREEN))
        self.wait(2)
        self.play(FadeOut(b07[1]))
        self.wait(2)
        self.play(FadeOut(vPoints), FadeIn(g_shp_norm))
        self.wait(2)
        self.play(FadeOut(x_input))
        
        self.wait(2)
        x_input2 = graph_time_series(ax_input, X[102,0], line_color=BLUE, add_vertex_dots=False)
        self.play(Create(x_input2))
        self.wait(2)
        shp_dist_norm0 = shapelet_dist_vect(shp, X[102,0], normalize=True, d=3)
        
        self.play(FadeOut(x_input2))
        ax_dist = Axes(
            x_range=[0, shp_dist_norm0.shape[0], 20],
            y_range=[0, shp_dist_norm0.max(), 2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(3.2*RIGHT)
        g_shp_dist_norm0 = graph_time_series(ax_dist, shp_dist_norm0, line_color=BLUE, add_vertex_dots=False)
        
        self.play(FadeOut(ax_input))
        self.play(FadeIn(ax_dist))
        self.wait(2)
        self.play(Create(g_shp_dist_norm0))
        self.wait(2)
        


        
        p, bins = np.histogram(shp_dist_norm0, bins=10, density=True)
        points = (bins[:-1] + bins[1:])//2
        ax_p = Axes(
            x_range=[points.min(), points.max()+1, 2],
            y_range=[0, 0.3, 0.05],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(3.2*RIGHT)
        
        area_p = graph_time_series(ax_p, points, y=p, line_color=BLUE, add_vertex_dots=False)
        
        
        self.play(FadeOut(ax_dist), FadeOut(g_shp_dist_norm0), FadeIn(ax_p))
        self.play(Create(area_p))
        
        lp0 = Line(start=Dot(ax_p.c2p(np.percentile(shp_dist_norm0,5),-0.1)), end=Dot(ax_p.c2p(np.percentile(shp_dist_norm0,5),0.15)), color=YELLOW)
        lp1 = Line(start=Dot(ax_p.c2p(np.percentile(shp_dist_norm0,10),-0.1)), end=Dot(ax_p.c2p(np.percentile(shp_dist_norm0,10),0.15)), color=YELLOW)
        self.play(b5[1].animate.set_stroke_color(GREEN), b5[0].animate.set_fill(GREEN))
        self.wait(2)
        self.play(FadeOut(b5[1]), Create(lp0))
        self.wait(2)
        self.play(b10[1].animate.set_stroke_color(GREEN), b10[0].animate.set_fill(GREEN))
        self.wait(2)
        self.play(FadeOut(b10[1]), Create(lp1))
        self.wait(2)
        
        txt_th = Tex(r'$\lambda \in [P(P_0,d(S,X)), P(P_1,d(S,X))]$').scale(0.8).shift(0.5*UP)
        txt_th2 = Tex(r'$\lambda = 1.81$').scale(0.8).shift(0.5*UP)
        self.play(FadeIn(txt_th))
        self.wait(2)
        self.play(FadeOut(txt_th), FadeIn(txt_th2))
        self.wait(2)
        self.play(FadeOut(lp0),FadeOut(lp1),FadeOut(area_p),FadeOut(ax_p))
        self.wait(2)
        self.play(FadeIn(ax_dist), FadeIn(g_shp_dist_norm0))
        self.wait(2)
        c_th = np.zeros(shp_dist_norm0.shape[0])+1.81
        g_threshold = graph_time_series(ax_dist, c_th, stroke_color=YELLOW, add_vertex_dots=False)
        self.play(Create(g_threshold))
        self.wait(2)
        
        
        # How is threshold set 
        
        #Init random
        #Dilation
        #ShpOcc even if not discriminant per say, the number of occurence of a common shape in still a discriminative factor
        #Result time/acc vs SoTa
        #Generalization to Multi/Uneven
        
"""   