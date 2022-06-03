from manim import *
from convst.utils.dataset_utils import load_sktime_dataset
from convst.utils.shapelets_utils import generate_strides_1D
import numpy as np
import pandas as pd

"""

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
#           CUSTOM MOBJECTS            #
#                                      # 
########################################


class Dot_ix(Dot):
    def __init__(self, ix=None, **kwargs):
        self.ix = ix
        super().__init__(**kwargs)


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
        

def range_from_x(x, n_steps=8, min_to=None, max_to=None):
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
    if step < 1:
        step = np.round(step, decimals=1)
    else:
        
        step = int(np.round(step, decimals=0))
    
    return [v_min, v_max, step]

def get_problem_data(
        name='GunPoint', default_shp_len=0.1, index_0=None, index_1=None,
        x0_index=None, x1_index=None
    ):
    X, y, _ = get_input_data(name=name)
    if x0_index is None:
        x0_index = np.random.choice(np.where(y==0)[0])
    if x1_index is None:
        x1_index = np.random.choice(np.where(y==1)[0])
    x0 = X[x0_index,0]
    x1 = X[x1_index,0]
    
    if index_0 is None:
        l0 = int(x0.shape[0] * default_shp_len)
        i_shp_0 = np.random.choice(x0.shape[0] - l0)
        index_0 = np.arange(i_shp_0,i_shp_0+l0)
    shp0 = x0[index_0]
    
    if index_1 is None:
        l1 = int(x1.shape[0] * default_shp_len)
        i_shp_1 = np.random.choice(x1.shape[0] - l1)
        index_1 = np.arange(i_shp_1,i_shp_1+l1)
    shp1 = x1[index_1]
    #Fix x0 x1, make ax scale to them
    return X, y, x0, x1, shp0, shp1

def get_features(X, shp, normalize=False, d=1, threshold=10):
    X_mins = np.zeros((X.shape[0], 3))
    for i in range(X.shape[0]):
        d_vect = shapelet_dist_vect(shp, X[i,0], normalize=normalize, d=d)
        X_mins[i,0] = d_vect.min()
        X_mins[i,1] = d_vect.argmin()
        X_mins[i,2] = (d_vect<=threshold).sum()
    return X_mins

########################################
#                                      #
#       GRAPHIC HELPER FUNCTIONS       #
#                                      # 
########################################

def boxed_txt(txt):
    text = Tex(txt)
    framebox = SurroundingRectangle(text, buff = .1, stroke_color=GREY)
    return VGroup(text, framebox)

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
        tex_list, color=WHITE, orientation=DOWN, center=False, aligned_edge=LEFT, buff=None
    ):
    vg = VGroup()
    for i in range(len(tex_list)):
        if isinstance(tex_list[i], mobject.text.tex_mobject.Tex):
            t = tex_list[i]
        else:
            t = Tex(tex_list[i], color=color)
        vg.add(t)
    if buff is None:
        return vg.arrange(orientation, center=center, aligned_edge=aligned_edge)  
    else:
        return vg.arrange(orientation, center=center, aligned_edge=aligned_edge, buff=buff)  

    
def get_axis_from_x(
        x, x_label, y_label, y=None, min_to=None, max_to=None ,tips=False,
        color=GREEN, numbered=True, label_color=WHITE, **kwargs
    ):
    
    x_range = range_from_x(np.arange(x.shape[-1]))
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
    
    
    labels = ax.get_axis_labels(
        x_label=x_label, y_label=y_label
    ).set_color(label_color).scale(0.6)
    
    return VGroup(ax,labels)
    
def get_11_axes(
        x_ranges, y_ranges, 
        x_labels=[Tex(''), Tex('')],
        y_labels=[Tex(''), Tex('')],
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
        },
        **kwargs
    ).scale(0.9).shift(2*DOWN).scale(0.5).shift(2.9*LEFT)
    
    x_labels0 = ax0.get_x_axis_label(x_labels[0]).set_color(label_color).scale(0.6)
    y_labels0 = ax0.get_y_axis_label(y_labels[0]).set_color(label_color).scale(0.6).shift(0.5*LEFT)

    
    ax1 = Axes(
        x_range=x_ranges[1],
        y_range=y_ranges[1],
        tips=tips,
        axis_config={
            "include_numbers": numbered,
            "color":color,
        },
        **kwargs
    ).scale(0.9).shift(2*DOWN).scale(0.5).shift(3.2*RIGHT)
    x_labels1 = ax1.get_x_axis_label(x_labels[1]).set_color(label_color).scale(0.6)
    y_labels1 = ax1.get_y_axis_label(y_labels[1]).set_color(label_color).scale(0.6).shift(0.5*LEFT)
    
    return VGroup(ax0,x_labels0,y_labels0), VGroup(ax1,x_labels1,y_labels1)

def get_211_axes(
        x_ranges, y_ranges, 
        x_labels=[Tex(''), Tex(''), Tex('')],
        y_labels=[Tex(''), Tex(''), Tex('')],
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
    x_labels0 = ax0.get_x_axis_label(x_labels[0]).set_color(label_color).scale(0.6)
    y_labels0 = ax0.get_y_axis_label(y_labels[0]).set_color(label_color).scale(0.6).shift(0.5*LEFT)
    
    ax1 = Axes(
        x_range=x_ranges[1],
        y_range=y_ranges[1],
        tips=tips,
        axis_config={
            "include_numbers": numbered,
            "color":color,
            **kwargs
        },
    ).scale(0.9).shift(2*DOWN).shift(2.75*LEFT).scale(0.475)
    x_labels1 = ax1.get_x_axis_label(x_labels[1]).set_color(label_color).scale(0.6)
    y_labels1 = ax1.get_y_axis_label(y_labels[1]).set_color(label_color).scale(0.6).shift(0.5*LEFT)
    

    ax2 = Axes(
        x_range=x_ranges[2],
        y_range=y_ranges[2],
        tips=tips,
        axis_config={
            "include_numbers": numbered,
            "color":color,
            **kwargs
        },
    ).scale(0.9).shift(2*DOWN).shift(2.75*RIGHT).scale(0.475)
    x_labels2 = ax2.get_x_axis_label(x_labels[2]).set_color(label_color).scale(0.6)
    y_labels2 = ax2.get_y_axis_label(y_labels[2]).set_color(label_color).scale(0.6).shift(0.5*LEFT)
    
    return VGroup(ax0,x_labels0,y_labels0), VGroup(ax1,x_labels1,y_labels1), VGroup(ax2,x_labels2,y_labels2)
    
def get_numerotation(i):
    #Calibrate so it is leftmost to the title at the bar level
    return Tex('{}'.format(i)).scale(0.7)

def make_title(title, i):
    tt = Title(title)
    num = get_numerotation(i).next_to(tt, LEFT)
    return VGroup(tt, num)
    
def load_all_data(x0_index=2, x1_index=0, shp0_x_index=np.arange(30,50), shp1_x_index=np.arange(102,123)):
    X, y, x0, x1, shp0, shp1 = get_problem_data(
        index_0=shp0_x_index, index_1=shp1_x_index,
        x0_index=x0_index, x1_index=x1_index
    )
    shp0_norm = (shp0 - shp0.mean())/shp0.std()
    shp1_norm = (shp1 - shp1.mean())/shp1.std()
    # _shpid_sampleid
    dist_vect_shp_0_0 = shapelet_dist_vect(shp0, x0, normalize=False, d=1)
    dist_vect_shp_0_1 = shapelet_dist_vect(shp0, x1, normalize=False, d=1)
    norm_dist_vect_shp_0_0 = shapelet_dist_vect(shp0, x0, normalize=True, d=1)
    norm_dist_vect_shp_0_1 = shapelet_dist_vect(shp0, x1, normalize=True, d=1)
    
    dist_vect_shp_1_0 = shapelet_dist_vect(shp1, x0, normalize=False, d=1)
    dist_vect_shp_1_1 = shapelet_dist_vect(shp1, x1, normalize=False, d=1)
    norm_dist_vect_shp_1_0= shapelet_dist_vect(shp1, x0, normalize=True, d=1)
    norm_dist_vect_shp_1_1 = shapelet_dist_vect(shp1, x1, normalize=True, d=1)
    
    X_T_0 = get_features(X, shp0, normalize=False, d=1, threshold=10)
    X_T_1 = get_features(X, shp1, normalize=False, d=1, threshold=10)
    return (
        x0_index, x1_index, shp0_x_index, shp1_x_index, X, y, x0, x1, shp0, shp1,
        shp0_norm, shp1_norm, dist_vect_shp_0_0, dist_vect_shp_0_1, norm_dist_vect_shp_0_0,
        norm_dist_vect_shp_0_1, dist_vect_shp_1_0, dist_vect_shp_1_1, norm_dist_vect_shp_1_0,
        norm_dist_vect_shp_1_1, X_T_0, X_T_1
    )


class Slide1(Scene):
    def construct(self):
        (x0_index, x1_index, shp0_x_index, shp1_x_index, X, y, x0, x1, shp0, shp1,
        shp0_norm, shp1_norm, dist_vect_shp_0_0, dist_vect_shp_0_1, norm_dist_vect_shp_0_0,
        norm_dist_vect_shp_0_1, dist_vect_shp_1_0, dist_vect_shp_1_1, norm_dist_vect_shp_1_0,
        norm_dist_vect_shp_1_0, X_T_0, X_T_1) = load_all_data()

        ########################################
        #                                      #
        #          INTRODUCE SHAPELETS         #
        #                                      # 
        ########################################
        title = make_title("What are Shapelets ?", 1)
        
        ax = Axes(
            x_range=[0,x0.shape[0],20],
            y_range=[x0.min()*1.075, x1.max()*1.075, 0.3],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN
            },
        ).scale(0.9).shift(0.75*DOWN)
        
        
        graph_x0 = graph_time_series(ax, x0, add_vertex_dots=False, line_color=COLOR_0)
        graph_x1 = graph_time_series(ax, x1, add_vertex_dots=False, line_color=COLOR_1)
        graph_shp0 = graph_time_series(
            ax, shp0_x_index ,y=shp0, add_vertex_dots=False,
            line_color=COLOR_SHP, stroke_width=11
        )

        DataGrp = VGroup(*ax, *graph_x0, *graph_x1, graph_shp0)
        
        self.play(FadeIn(title))
        self.wait(1.5)
        self.play(Create(ax), run_time=3)
        self.play(Create(graph_x0), run_time=2.5)
        self.play(Create(graph_x1), run_time=2.5)
        self.wait(2)
        self.play(Create(graph_shp0),run_time=2)
        self.play(DataGrp.animate.stretch(1.8, 0).shift(1.25*UP).scale(0.55))
        self.wait(2)
        self.play(graph_shp0.animate.shift(1.5*DOWN).shift(2.45*LEFT))
        
        Tex_shp = Tex(
            r' A Shapelet $S = \{s_1, \ldots, s_l\}$, with $l$ the length parameter'
        ).scale(0.8).next_to(graph_shp0, RIGHT)
        
        self.play(FadeIn(Tex_shp))
        self.wait()
        
        
       
class Slide2(Scene):
    def construct(self):
        (x0_index, x1_index, shp0_x_index, shp1_x_index, X, y, x0, x1, shp0, shp1,
        shp0_norm, shp1_norm, dist_vect_shp_0_0, dist_vect_shp_0_1, norm_dist_vect_shp_0_0,
        norm_dist_vect_shp_0_1, dist_vect_shp_1_0, dist_vect_shp_1_1, norm_dist_vect_shp_1_0,
        norm_dist_vect_shp_1_0, X_T_0, X_T_1) = load_all_data()
        ########################################
        #                                      #
        #       SHOW HOW TO EXTRACT MIN        #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("How are Shapelets used ?", 2)
        vg_all.add(*title)
        
        x_ranges = [
            [0,x0.shape[0],20],
            [0,dist_vect_shp_1_0.shape[0],20],
            range_from_x(X_T_0[:,0])
        ]
        
        y_ranges = [
            [-1.0,2.0,0.5],
            [0, max(dist_vect_shp_0_0.max(),dist_vect_shp_1_0.max()), 10],
            range_from_x(X_T_1[:,0])
        ]
        
        x_labels = [
            Tex(''), Tex(''), Tex(r"$\min d({\cal X},S_1)$")
        ]
        
        y_labels = [
            Tex(''), Tex(''), Tex(r"$\min d({\cal X},S_2)$")
        ]
        
        ax0, ax1, ax2 = get_211_axes(
            x_ranges, y_ranges, x_labels=x_labels, y_labels=y_labels
        )
        
        graph_x0 = graph_time_series(ax0[0], x0, add_vertex_dots=False, line_color=COLOR_0)
        graph_x1 = graph_time_series(ax0[0], x1, add_vertex_dots=False, line_color=COLOR_1)
        d_vect_0_1 = graph_time_series(
            ax1[0], dist_vect_shp_0_1,
            line_color=COLOR_1, add_vertex_dots=False
        )
        graph_shp1 = graph_time_series(
            ax0[0], shp1_x_index, y=shp1, stroke_width=11,
            line_color=COLOR_SHP, add_vertex_dots=False
        )
        graph_shp0 = graph_time_series(
            ax0[0], shp0_x_index ,y=shp0, add_vertex_dots=False,
            line_color=COLOR_SHP, stroke_width=11
        )
        vg_all.add(*[ax0, ax1, ax2, graph_x0, graph_x1, d_vect_0_1,
                     graph_shp1, graph_shp0])
        t = ValueTracker(0)
        x_space = np.arange(dist_vect_shp_0_0.shape[0])
        
        def update_shp():
            i = int(t.get_value())
            graph_shp = graph_time_series(
                ax0[0], np.arange(i,i+shp0.shape[0]), y=shp0,
                line_color=COLOR_SHP,
                add_vertex_dots=False,
                stroke_width=10
            )
            return graph_shp
        
        def get_shp_line_dots():
            step = 1
            i = int(t.get_value())
            vg = VGroup()
            for st in range(0,shp0.shape[0],step):
                yp0 = min(shp0[st],x0[i+st])
                yp1 = max(shp0[st],x0[i+st])
                p1 = Dot(ax0[0].c2p(i+st, yp0))
                p2 = Dot(ax0[0].c2p(i+st, yp1))
                vg.add(DashedLine(start=p1.get_center(), end=p2.get_center(), stroke_color=YELLOW))
            return vg

        def get_dist_vect():
            i = int(t.get_value())
            return graph_time_series(ax1[0], dist_vect_shp_0_0[:i+1], line_color=COLOR_0, add_vertex_dots=False)
                

        l_dots = always_redraw(get_shp_line_dots)
        graph_shp = always_redraw(update_shp)
        d_vect = always_redraw(get_dist_vect)
        
        d_formula = Tex_Group(
            [r'$d(X,S) = \{v_1, \ldots, v_{m-(l-1)}\}$',
             r'with $v_i = \sum_{j=1}^l |x_{i+(j-1)} - s_j|$']
        ).scale(0.75).next_to(ax1,RIGHT).shift(0.7*RIGHT)
        min0 = SurroundingRectangle(Dot(ax1[0].c2p(dist_vect_shp_0_0.argmin(), dist_vect_shp_0_0.min()))).scale(0.65)
        min1 = SurroundingRectangle(Dot(ax1[0].c2p(dist_vect_shp_0_1.argmin(), dist_vect_shp_0_1.min()))).scale(0.65)
        
        dots0 = Dot(ax2[0].c2p(X_T_0[x0_index,0],X_T_1[x0_index,0]), fill_color=COLOR_0).scale(0.85)
        dots1 = Dot(ax2[0].c2p(X_T_0[x1_index,0],X_T_1[x1_index,0]), fill_color=COLOR_1).scale(0.85)
                                                
        all_dots = VGroup()
        for i in range(X.shape[0]):
            if i != x0_index and i != x1_index:
                if y[i] == y[x0_index]:
                    all_dots.add(Dot(ax2[0].c2p(X_T_0[i,0],X_T_1[i,0]), fill_color=COLOR_0, fill_opacity=0.75).scale(0.55))
                else:
                    all_dots.add(Dot(ax2[0].c2p(X_T_0[i,0],X_T_1[i,0]), fill_color=COLOR_1, fill_opacity=0.75).scale(0.55))
        
        min0_tex = Tex(r'$S_1$', color=PURPLE, stroke_width=1.5).scale(0.65)
        min0_tex.next_to(graph_shp0, DOWN).shift(UP*0.15)
        min1_tex = Tex(r'$S_2$', color=PURPLE, stroke_width=1.5).scale(0.65)
        min1_tex.next_to(graph_shp1, UP).shift(DOWN*0.75)
        vg_all.add(*[d_vect, min0, 
                     min1, dots0, dots1, all_dots, min0_tex, min1_tex])
        
        
        # TODO : Add a d(S1, X) to ax1 to know what we are graphing
        self.play(FadeIn(title))
        self.play(Create(ax0), Create(graph_x0))
        self.play(Create(ax1))
        self.play(Create(graph_shp), Create(l_dots), Create(d_vect))
        self.wait(1)
        self.play(FadeIn(d_formula))
        self.wait(4)
        self.play(t.animate.set_value(x_space[-1]), run_time=10, rate_func=rate_functions.ease_in_sine)
        self.wait(1)
        self.play(FadeOut(graph_shp), FadeOut(l_dots))
        self.play(Create(graph_x1))
        self.play(Create(d_vect_0_1))
        self.wait(1)
        self.play(Create(min0), Create(min1))
        self.wait(1)
        self.play(FadeOut(d_formula), Create(ax2))
        self.wait(1)
        self.play(Create(graph_shp1), Create(graph_shp0))
        self.play(FadeIn(min0_tex), FadeIn(min1_tex))
        self.wait(1)
        self.play(Transform(min0, dots0), Transform(min1, dots1))
        self.wait(1)
        self.play(Create(all_dots), run_time=3)
        self.wait(1)
        
        
class Slide3(Scene):
    def construct(self):
        (x0_index, x1_index, shp0_x_index, shp1_x_index, X, y, x0, x1, shp0, shp1,
        shp0_norm, shp1_norm, dist_vect_shp_0_0, dist_vect_shp_0_1, norm_dist_vect_shp_0_0,
        norm_dist_vect_shp_0_1, dist_vect_shp_1_0, dist_vect_shp_1_1, norm_dist_vect_shp_1_0,
        norm_dist_vect_shp_1_0, X_T_0, X_T_1) = load_all_data()
        ########################################
        #                                      #
        #         SHOW ARGMIN Z-Norm           #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Improvements of the base formulation ", 3)
        vg_all.add(*title)
        #Show znorm first and argmin on distance vector

        x_ranges = [
            [0,x0.shape[0],20],
            [0,dist_vect_shp_1_0.shape[0],20],
            [0, shp0_norm.shape[0], 2]
        ]
        
        y_ranges = [
            [-1.0,2.0,0.5],
            [0, max(norm_dist_vect_shp_0_0.max(),norm_dist_vect_shp_0_1.max()), 10],
            [min(shp0_norm.min(),shp1_norm.min()),max(shp0_norm.max(),shp1_norm.max(), 0.5)]
        ]
                
        ax0, ax1, ax2 = get_211_axes(x_ranges, y_ranges)
        
        vg_all.add(ax1)
        t = ValueTracker(0)
        x_space = np.arange(norm_dist_vect_shp_0_0.shape[0])
        graph_x0 = graph_time_series(ax0[0], x0, add_vertex_dots=False, line_color=COLOR_0)
        graph_x1 = graph_time_series(ax0[0], x1, add_vertex_dots=False, line_color=COLOR_1)
     
        
        
        def get_dist_vect():
            i = int(t.get_value())
            return graph_time_series(ax1[0], norm_dist_vect_shp_0_0[:i+1], stroke_color=COLOR_0, add_vertex_dots=False)
                
        def x_subseq_loc():
            i = int(t.get_value())
            length = shp0_norm.shape[0]-1
            x_sub = x0[i:i+shp0_norm.shape[0]]
            vg = VGroup()
            vg.add(Dot(ax0[0].c2p(i, x_sub[0])))
            vg.add(Dot(ax0[0].c2p(i+length, x_sub[length])))
            return SurroundingRectangle(vg, buff=0.01)
        
        def x_subseq():
            i = int(t.get_value())
            x_sub = x0[i:i+shp0_norm.shape[0]]
            x_sub_norm = (x_sub - x_sub.mean()) / x_sub.std()
            return graph_time_series(ax2[0], x_sub_norm, stroke_color=COLOR_0, add_vertex_dots=False)
        
        d_vect = always_redraw(get_dist_vect)
        x_subs = always_redraw(x_subseq)
        
        x_subs_loc = always_redraw(x_subseq_loc)
        vg_all.add(d_vect)
        z_norm_txt = Tex_Group(
            [r'$S_{norm} = (S - mean(S)) / std(S)$',
             r'$X_{norm} = (X - mean(X)) / std(X)$']
        ).next_to(ax2[0], 0.5*LEFT).scale(0.65).shift(0.825*RIGHT)
        
        g_shp0_normed = graph_time_series(ax2[0], shp0_norm, stroke_color=COLOR_SHP, add_vertex_dots=False, stroke_width=10)
        g_shp_dist_norm1 = graph_time_series(ax1[0], norm_dist_vect_shp_0_1, stroke_color=COLOR_1, add_vertex_dots=False)
        vg_all.add(g_shp_dist_norm1)
        min0 = SurroundingRectangle(Dot(ax1[0].c2p(norm_dist_vect_shp_0_0.argmin(), norm_dist_vect_shp_0_0.min()))).scale(0.65)
        min1 = SurroundingRectangle(Dot(ax1[0].c2p(norm_dist_vect_shp_0_1.argmin(), norm_dist_vect_shp_0_1.min()))).scale(0.65)
        
        
        x_ranges[-1] = [X_T_0[:,1].min(), X_T_0[:,1].max()+40, 20]
        y_ranges[-1] = [X_T_0[:,0].min(), X_T_0[:,0].max()+5, 5]
        
        x_labels = [
            Tex(''), Tex(''), Tex(r"arg$\min d({\cal X},S)$")
        ]
        
        y_labels = [
            Tex(''), Tex(''), Tex(r"$\min d({\cal X},S)$")
        ]
        _, _, ax2_points = get_211_axes(
            x_ranges, y_ranges, x_labels=x_labels, y_labels=y_labels
        )
        
        dots0 = Dot(ax2_points[0].c2p(X_T_0[x0_index,1],X_T_0[x0_index,0]), fill_color=COLOR_0).scale(0.85)
        dots1 = Dot(ax2_points[0].c2p(X_T_0[x1_index,1],X_T_0[x1_index,0]), fill_color=COLOR_1).scale(0.85)
        all_dots = VGroup()
        for i in range(X.shape[0]):
            if i != x0_index and i != x1_index:
                if y[i] == y[x0_index]:
                    all_dots.add(Dot(ax2_points[0].c2p(X_T_0[i,1],X_T_0[i,0]), fill_color=COLOR_0, fill_opacity=0.75).scale(0.55))
                else:
                    all_dots.add(Dot(ax2_points[0].c2p(X_T_0[i,1],X_T_0[i,0]), fill_color=COLOR_1, fill_opacity=0.75).scale(0.55))
        vg_all.add(all_dots)
        vg_all.add(*[g_shp_dist_norm1,min0,min1,ax0,graph_x0,all_dots,dots1,dots0,ax2_points,graph_x1])
        self.play(FadeIn(title))
        self.wait(2)
        self.play(Create(ax0), Create(graph_x0))
        self.play(Create(ax2))
        self.wait()
        self.play(Create(d_vect),Create(g_shp0_normed),Create(x_subs),Create(x_subs_loc))
        self.wait(3)
        self.play(FadeIn(z_norm_txt))
        self.wait(6)
        self.play(FadeOut(z_norm_txt), Create(ax1))
        self.play(t.animate.set_value(x_space[-1]), run_time=14, rate_func=rate_functions.ease_in_sine)
        self.wait()
        self.play(FadeOut(x_subs_loc))
        self.wait()
        self.play(Create(graph_x1), Create(g_shp_dist_norm1))
        self.wait()
        self.play(FadeOut(ax2), FadeOut(x_subs), FadeOut(g_shp0_normed))
        self.play(FadeIn(ax2_points))
        self.wait()
        self.play(Create(min0), Create(min1))
        self.wait()
        self.play(Transform(min0, dots0), Transform(min1, dots1))
        self.wait()
        self.play(Create(all_dots), run_time=5)
        self.wait()
        

class Slide4(Scene):
    def construct(self):
        ########################################
        #                                      #
        #       SHOW Shapelet Occurence        #
        #                                      # 
        ########################################
        (x0_index, x1_index, shp0_x_index, shp1_x_index, X, y, x0, x1, shp0, shp1,
        shp0_norm, shp1_norm, dist_vect_shp_0_0, dist_vect_shp_0_1, norm_dist_vect_shp_0_0,
        norm_dist_vect_shp_0_1, dist_vect_shp_1_0, dist_vect_shp_1_1, norm_dist_vect_shp_1_0,
        norm_dist_vect_shp_1_0, X_T_0, X_T_1) = load_all_data()
        vg_all = VGroup()
        title = make_title("Contributions : Shapelet Occurence", 5)
        vg_all.add(title)
        #Anim threshold and show effect on points with anim
        x_ranges = [
            [0, norm_dist_vect_shp_0_0.shape[0], 20],
            [0, norm_dist_vect_shp_0_0.shape[0]+20, 20]
            
        ]
        y_ranges = [
            [norm_dist_vect_shp_0_0.min(), norm_dist_vect_shp_0_0.max(), 10],
            [0, 100, 10]
            
        ]
        
        x_labels = [
            Tex(''), Tex('arg$\min d(S,{\cal X})$')
        ]

        y_labels = [
            Tex(''), Tex('$d(S,{\cal X}) <= \lambda$')
        ]
        
        # TODO : Will need to reshape ax1 2 and adjust ranges
        ax1, ax2 = get_11_axes(
            x_ranges, y_ranges,
            x_labels=x_labels,
            y_labels=y_labels
            #y_axis_config={"scaling": LogBase(custom_labels=True)},
        )
        ax1.shift(UP)
        ax2.shift(UP)
            
        vg_all.add(*[ax1,ax2])
        t = ValueTracker(0.5)
        x_space = np.arange(0.5,20.5,2)

        
        def update_threshold_area():
            i_na = float(t.get_value())
            vg = VGroup()
            c_th = np.zeros(norm_dist_vect_shp_0_0.shape[0]) + i_na
            x_area = np.where(norm_dist_vect_shp_0_0 <= i_na)[0]
            
            x_areas = []
            prec = None
            area = []
            for j in range(len(x_area)):
            
                if prec is None or x_area[j] == prec + 1:
                    area.append(x_area[j])
                    prec = x_area[j]
                else:
                    x_areas.append(area)
                    area = [x_area[j]]
                    prec = x_area[j]
            x_areas.append(area)
            
            for j in range(len(x_areas)):
                x_area = x_areas[j]
                if len(x_area)>1:
                    y_area = norm_dist_vect_shp_0_0[x_area]
                    y_th = c_th[x_area]
                    l1 = []
                    l2 = []
                    for i in range(len(x_area)):
                        l1.append(ax1[0].c2p(x_area[i], y_area[i]))
                        l2.append(ax1[0].c2p(x_area[i], y_th[i]))
                    l_concat = l1 + l2[::-1]
                    vg.add(Polygon(*l_concat, color=GREY_B, fill_opacity=0.85, stroke_width=0))
            return vg
            
        
        def update_threshold():
            i = float(t.get_value())
            vg = VGroup()
            c_th = np.zeros(norm_dist_vect_shp_0_0.shape[0]) + i
            g = graph_time_series(ax1[0], c_th, stroke_color=COLOR_SHP, add_vertex_dots=False)
            txt = Tex('$\lambda$ threshold').next_to(g,UP).scale(0.6).shift(RIGHT).shift(0.25*DOWN)
            vg.add(*[g, txt])
            return vg

    
        def get_dot(i, ix):
            X_mins = get_features(X[ix:ix+1], shp0, normalize=True, d=1, threshold=i)
            return ax2[0].c2p(X_mins[0,1],X_mins[0,2])
          
        def get_dots(i):
            X_mins = get_features(X, shp0, normalize=True, d=1, threshold=i)
            dot_list = []
            for ix in range(X.shape[0]):
                if y[ix] == y[x0_index]:
                    dot = Dot_ix(ix=ix, point=ax2[0].c2p(X_mins[0,1],X_mins[0,2]), fill_color=COLOR_0, fill_opacity=0.75).scale(0.55)
                else:
                    dot = Dot_ix(ix=ix, point=ax2[0].c2p(X_mins[0,1],X_mins[0,2]), fill_color=COLOR_1, fill_opacity=0.75).scale(0.55)
                dot.add_updater(lambda x: x.move_to(get_dot(float(t.get_value()),x.ix)))
                dot_list.append(dot)
            return dot_list
        
        
            
            
        g_shp_dist_norm0 = graph_time_series(ax1[0], norm_dist_vect_shp_0_0, stroke_color=COLOR_0, add_vertex_dots=False)    
        vg_all.add(g_shp_dist_norm0)
        area_threshold = always_redraw(update_threshold_area)
        threshold = always_redraw(update_threshold)
        dots_th = get_dots(0.5)
        self.play(FadeIn(title))
        self.wait(2)
        self.play(Create(ax1),Create(ax2))        
        self.play(Create(g_shp_dist_norm0))
        self.wait(2)
        self.play(Create(area_threshold))
        self.play(Create(threshold))
        for i in range(len(dots_th)):
            self.add(dots_th[i])
        self.wait(1)
        self.play(t.animate.set_value(x_space[-1]), run_time=6, rate_func=rate_functions.ease_in_sine)
        self.play(t.animate.set_value(x_space[0]), run_time=6, rate_func=rate_functions.ease_in_sine)
        self.play(t.animate.set_value(x_space[-1]), run_time=6, rate_func=rate_functions.ease_in_sine)
        self.play(t.animate.set_value(x_space[0]), run_time=6, rate_func=rate_functions.ease_in_sine)
        self.wait()
        

class Slide5(Scene):
    def construct(self):
        ########################################
        #                                      #
        #        SHOW Dilated Shapelets        #
        #                                      # 
        ########################################
        (x0_index, x1_index, shp0_x_index, shp1_x_index, X, y, x0, x1, shp0, shp1,
        shp0_norm, shp1_norm, dist_vect_shp_0_0, dist_vect_shp_0_1, norm_dist_vect_shp_0_0,
        norm_dist_vect_shp_0_1, dist_vect_shp_1_0, dist_vect_shp_1_1, norm_dist_vect_shp_1_0,
        norm_dist_vect_shp_1_0, X_T_0, X_T_1) = load_all_data(shp0_x_index=np.arange(35,45))
        
        vg_all = VGroup()
        title = make_title("Contributions : Dilated Shapelet", 6)
        vg_all.add(*title)
        
        
        
        x_ranges = [
            [0,x0.shape[0],20],
            [0, norm_dist_vect_shp_0_0.shape[0]+10,20],
            [0, norm_dist_vect_shp_0_0.shape[0], 20]
        ]
        
        y_ranges = [
            [-1.0,2.0,0.5],
            [0, max(norm_dist_vect_shp_0_0.max(),norm_dist_vect_shp_0_1.max()), 5],
            [0, norm_dist_vect_shp_0_0.shape[0]//2, 10]
        ]
            
        x_labels = [
            Tex(''),Tex(''), Tex('arg$\min d(S,{\cal X})$')
        ]

        y_labels = [
            Tex(''),Tex('normalized distance vector'), Tex('$d(S,{\cal X}) <= \lambda$')
        ]
        
        
        ax0, ax1, ax2 = get_211_axes(x_ranges, y_ranges, x_labels=x_labels, y_labels=y_labels)
        ax1.shift(0.5*DOWN)
        ax2.shift(0.5*DOWN)
        ax2[2].shift(0.35*DOWN)
        graph_x0 = graph_time_series(ax0[0], x0, add_vertex_dots=False, line_color=COLOR_0)
        #Check if the previous one are OK
        self.play(FadeIn(title))
        self.play(Create(ax0), Create(ax1), Create(ax2))
        self.play(Create(graph_x0))
        c_th = np.zeros(norm_dist_vect_shp_0_0.shape[0]) + 5
        g = graph_time_series(ax1[0], c_th, stroke_color=COLOR_SHP, add_vertex_dots=False)
        self.play(Create(g))
        vg_all.add(*[graph_x0, ax0, ax1, ax2, g])
        for d in [1,3,5,7]:
            shp_dist_norm0 = shapelet_dist_vect(shp0, x0, normalize=True, d=d)
            t = ValueTracker(0)
            
            # Tracker functions
            
            def update_shp():
                i = int(t.get_value())
                vg = VGroup()
                for j in range(shp0.shape[0]):
                    vg.add(Dot(ax0[0].c2p(i+(j*d), shp0[j]),fill_color=COLOR_SHP).scale(0.8))
                return vg
            
            def get_shp_line_dots():
                step = 1
                i = int(t.get_value())
                vg = VGroup()
                for st in range(0,shp0.shape[0],step):
                    yp0 = min(shp0[st],x0[i+(d*st)])
                    yp1 = max(shp0[st],x0[i+(d*st)])
                    p1 = Dot(ax0[0].coords_to_point(i+(d*st), yp0))
                    p2 = Dot(ax0[0].coords_to_point(i+(d*st), yp1))
                    vg.add(DashedLine(start=p1.get_center(), end=p2.get_center(), stroke_color=YELLOW))
                return vg
    
            def get_dist_vect():
                i = int(t.get_value())
                return graph_time_series(ax1[0], shp_dist_norm0[:i+1], line_color=COLOR_0, add_vertex_dots=False)
                    
            
            def brace_d():
                i = int(t.get_value())
                vg = VGroup(
                    Dot(ax0[0].c2p(i+0.5, shp0[0]+0.15)),
                    Dot(ax0[0].c2p(i+d-0.5, shp0[1]+0.15))
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
            self.play(t.animate.set_value(x_space[-1]), run_time=5, rate_func=rate_functions.ease_in_sine)
            
            X_mins = get_features(X, shp0, normalize=True, d=d, threshold=5)
            
            all_dots = VGroup()
            for i in range(X.shape[0]):
                if y[i] == y[x0_index]:
                    all_dots.add(Dot(ax2[0].c2p(X_mins[i,1],X_mins[i,2]), fill_color=COLOR_0, fill_opacity=0.75).scale(0.55))
                else:
                    all_dots.add(Dot(ax2[0].c2p(X_mins[i,1],X_mins[i,2]), fill_color=COLOR_1, fill_opacity=0.75).scale(0.55))
            self.play(FadeIn(all_dots))
            self.wait(2)
            if d < 7:
                self.play(FadeOut(l_dots),FadeOut(graph_shp),FadeOut(d_vect), FadeOut(all_dots), FadeOut(brace))
        
        self.wait()

class Slide6(Scene):
    def construct(self):
        ########################################
        #                                      #
        #              SHOW RDST               #
        #                                      # 
        ########################################
        (x0_index, x1_index, shp0_x_index, shp1_x_index, X, y, x0, x1, shp0, shp1,
        shp0_norm, shp1_norm, dist_vect_shp_0_0, dist_vect_shp_0_1, norm_dist_vect_shp_0_0,
        norm_dist_vect_shp_0_1, dist_vect_shp_1_0, dist_vect_shp_1_1, norm_dist_vect_shp_1_0,
        norm_dist_vect_shp_1_0, X_T_0, X_T_1) = load_all_data()
        
        vg_all = VGroup()
        title = make_title("Contributions : Random Dilated Shapelet Transform", 7)
        vg_all.add(*title)
        
        
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
        
        self.play(FadeIn(title))
        self.wait(15)
        self.play(Create(g_length))
        self.wait(4)
        self.play(g_length.animate.scale(0.8).shift(5*LEFT).shift(1.75*UP))
        self.wait()
        self.play(Create(g_thresd))
        self.wait(2)
        self.play(g_thresd.animate.scale(0.8).shift(2.5*LEFT).shift(1.75*UP))
        self.wait()
        self.play(Create(g_pnorm))
        self.wait(2)
        self.play(g_pnorm.animate.scale(0.8).shift(1.75*UP))
        self.wait()
        self.play(Create(txt_in))
        self.wait(2)
        self.play(txt_in.animate.scale(0.8).shift(2.5*RIGHT).shift(1.75*UP))
        self.add(txt_in_T.scale(0.8).shift(2.5*RIGHT).shift(1.75*UP))
        self.wait()
        self.play(Create(txt_n_shp))
        self.wait(2)
        self.play(txt_n_shp.animate.scale(0.8).shift(5*RIGHT).shift(1.75*UP))
        self.wait()
        
        
        
        idx_shp = [105,109,113,117,121,125,129]
        shp = x0[idx_shp]
        shp_norm = (shp - shp.mean()) / shp.std()
        
        x_ranges = [
            [0, shp.shape[0], 1],
            [0, x0.shape[0], 20],
        ]
        y_ranges = [
            [shp_norm.min(), shp_norm.max(), 0.5],
            [x0.min(), x0.max(), 0.5],
        ]

        ax1, ax2 = get_11_axes(
            x_ranges, y_ranges,
        )
        
        Tx_algo = Tex('Choose a random length').scale(0.8)
        x_input = graph_time_series(ax2[0], x0, line_color=COLOR_0, add_vertex_dots=False)
        
        self.play(FadeIn(Tx_algo), b7[1].animate.set_stroke_color(GREEN), b7[0].animate.set_fill(GREEN))
        self.play(Transform(b7[1],ax1[0]), )
        self.wait(2.5)
        self.play(txt_in_T.animate.set_fill(GREEN), FadeOut(Tx_algo))
        Tx_algo = Tex('Choose a random sample').scale(0.8)
        self.play(FadeIn(Tx_algo),FadeOut(txt_in_T),FadeIn(ax2[0]), FadeIn(x_input))
        self.wait(2.5)
        self.play(FadeOut(Tx_algo))
        txt_dil = Tex(r'd = $\lfloor 2^{x} \rfloor$, $x \in [0, log \frac{||X||}{l}]$').scale(0.8).shift(0.5*UP)
        txt_dil1 = Tex(r'd = 4').scale(0.8).shift(0.5*UP).shift(5*LEFT)
        self.play(Create(txt_dil))
        self.wait(5)
        self.play(Transform(txt_dil, txt_dil1))
        self.wait(2.5)
        txt_point = Tex(r'$ i \in [0, ||X|| - (l-1)\times d]$').scale(0.8).shift(0.5*UP)
        txt_point3 = Tex(r'$ i = 105 $').scale(0.8).shift(0.5*UP).shift(5*RIGHT)
        self.play(Create(txt_point))
        self.wait(2.5)
        self.play(Transform(txt_point, txt_point3))
        self.wait(2.5)
        vPoints = VGroup()
        for ix in idx_shp:
            vPoints.add(Dot(ax2[0].c2p(ix, x0[ix]),fill_color=PURPLE).scale(0.9))
        self.play(Create(vPoints, run_time=5, rate_func=rate_functions.ease_in_sine))
        self.wait()
        shp_norm = (shp - shp.mean()) / shp.std()
        g_shp_norm = graph_time_series(ax1[0], shp_norm, line_color=PURPLE, add_vertex_dots=False)
        self.play(b07[1].animate.set_stroke_color(GREEN), b07[0].animate.set_fill(GREEN))
        self.wait(2)
        self.play(FadeOut(b07[1]))
        self.wait(2)
        self.play(FadeOut(vPoints), FadeIn(g_shp_norm))
        self.wait(2)
        Tx_algo = Tex('Choose another sample of the same class').scale(0.8)
        self.play(FadeOut(x_input), FadeIn(Tx_algo))
        
        self.wait(2)
        x_input2 = graph_time_series(ax2[0], X[92,0], line_color=COLOR_0, add_vertex_dots=False)
        self.play(Create(x_input2))
        self.wait(2)
        shp_dist_norm0 = shapelet_dist_vect(shp, X[92,0], normalize=True, d=4)
        self.play(FadeOut(x_input2), FadeOut(ax2),FadeOut(Tx_algo))
        ax_dist = Axes(
            x_range=[0, shp_dist_norm0.shape[0], 20],
            y_range=[0, shp_dist_norm0.max(), 2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "color":GREEN         
            },
        ).scale(0.9).shift(2*DOWN).scale(0.5).shift(3.2*RIGHT)
        Tx_algo = Tex('Compute the distance vector').scale(0.8)
        g_shp_dist_norm0 = graph_time_series(ax_dist, shp_dist_norm0, line_color=COLOR_0, add_vertex_dots=False)
        
        self.play(FadeIn(ax_dist),FadeIn(Tx_algo))
        self.wait(2)
        self.play(Create(g_shp_dist_norm0))
        self.wait(2)
    
        
        lp0 = Line(start=Dot(ax_dist.c2p(0,np.percentile(shp_dist_norm0,5))), end=Dot(ax_dist.c2p(shp_dist_norm0.shape[0], np.percentile(shp_dist_norm0,5))), color=YELLOW)
        lp1 = Line(start=Dot(ax_dist.c2p(0,np.percentile(shp_dist_norm0,10))), end=Dot(ax_dist.c2p(shp_dist_norm0.shape[0], np.percentile(shp_dist_norm0,10))), color=YELLOW)
        self.play(FadeOut(Tx_algo), b5[1].animate.set_stroke_color(GREEN), b5[0].animate.set_fill(GREEN))
        self.wait(2)
        self.play(FadeOut(b5[1]), Create(lp0))
        self.wait(2)
        self.play(b10[1].animate.set_stroke_color(GREEN), b10[0].animate.set_fill(GREEN))
        self.wait(2)
        self.play(FadeOut(b10[1]), Create(lp1))
        self.wait(2)
        th = np.round(np.random.choice(np.arange(np.percentile(shp_dist_norm0,7.5), np.percentile(shp_dist_norm0,10), 0.01)),decimals=2)
        
        
        txt_th = Tex(r'$\lambda \in [P(P_0,d(S,X)), P(P_1,d(S,X))]$').scale(0.8).shift(0.5*UP)
        txt_th2 = Tex(r'$\lambda = {}$'.format(th)).scale(0.8).shift(0.5*UP)
        self.play(FadeIn(txt_th))
        self.wait(2)
        self.play(FadeOut(txt_th), FadeIn(txt_th2))
        self.wait(2)
        self.play(FadeOut(lp0),FadeOut(lp1))
        self.wait(2)
        c_th = np.zeros(shp_dist_norm0.shape[0])+th
        g_threshold = graph_time_series(ax_dist, c_th, stroke_color=YELLOW, add_vertex_dots=False)
        self.play(Create(g_threshold))
        self.wait(2)
        
        

class Slide7(Scene):
    def construct(self):
        ########################################
        #                                      #
        #           TIMING Results             #
        #                                      # 
        ########################################
        res_path = '~\\Documents\\git_projects\\convst\\results\\'
        sample_bench = pd.read_csv(res_path+'n_samples_benchmarks.csv', index_col=0)
        length_bench = pd.read_csv(res_path+'n_timepoints_benchmarks.csv', index_col=0)
        
        vg_all = VGroup()
        title = make_title("Time Complexity", 8)
        vg_all.add(*title)
        
        
        x_ranges = [
            [225, 1800, 225],
            [355, 2840, 355],
        ]
        y_ranges = [
            [-0.6, 5, 1],
            [-0.6, 5, 1],
        ]

        y_labels = [
            Tex('Time series length',font_size=65), Tex(r"Number of samples",font_size=65)
        ]
        ax1, ax2 = get_11_axes(
            x_ranges, y_ranges,
            y_labels=y_labels,
            y_axis_config={"scaling": LogBase(custom_labels=True), 'include_numbers':False},
            x_axis_config={"font_size":55}
        )
        ax1[2].shift(0.65*LEFT)
        ax2[2].shift(0.65*LEFT)
        ax1[0][1].add_labels({
            6*(10**-1):'600ms',
            6*(10**0):'6s',
            6*(10**1):'1min',
            6*(10**2):'10min',
            6*(10**3):'1h40',
            6*(10**4):'16h40',
            },font_size=24
        )
        ax1.shift(2.75*UP)
        ax2.shift(2.75*UP)
        ax1[2].shift(RIGHT)
        ax2[2].shift(RIGHT)
        v_samples = VGroup()
        v_length = VGroup()
        color_dict = {
            'Rocket': GREEN,
            'RDST': PURPLE,
            'STC':BLUE,
            'DrCIF' : DARK_BROWN,
            'HC1':ORANGE,
            'HC2':RED
        }
        txc = Tex_Group([Tex(name, color=c) for name, c in color_dict.items()], orientation=RIGHT, buff=1).shift(1.65*DOWN).shift(3.75*LEFT)
        
        
        for model in ['RDST', 'Rocket', 'DrCIF', 'HC1', 'HC2', 'STC']:
            sw = 4
            if model == 'RDST':
                sw=8
            
            g = graph_time_series(
                ax1[0], sample_bench.index.values,
                y=sample_bench[model].values,
                stroke_color=color_dict[model],
                add_vertex_dots=False, stroke_width=sw
            )
            v_samples.add(g)
            
            g = graph_time_series(
                ax2[0], length_bench.index.values,
                y=length_bench[model].values, 
                stroke_color=color_dict[model],
                add_vertex_dots=False, stroke_width=sw
            )
            v_length.add(g)
        vg_all.add(*[v_samples, v_length, ax1, ax2])
        
        self.play(FadeIn(title))
        self.wait()
        self.play(Create(ax1), Create(ax2))
        self.play(FadeIn(txc))
        self.wait()
        self.play(Create(v_samples), Create(v_length))
        self.wait()
        # TODO : Will need to reshape ax1 2 and adjust ranges
        
        
class Slide8(Scene):
    def construct(self):
        ########################################
        #                                      #
        #              ACC Results             #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Accuracy Results", 9)
        vg_all.add(*title)
        self.play(FadeIn(title))
        
        
        
        self.wait()
        self.play(FadeOut(vg_all))
        self.wait()
        
        #Parameter sensitivity if time
        
class Slide9(Scene):
    def construct(self):
        ########################################
        #                                      #
        #             Conclusion               #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Conclusion \& Future Works", 10)
        vg_all.add(*title)
        self.play(FadeIn(title))
        
        
        
        self.wait()
        self.play(FadeOut(vg_all))
        self.wait()
