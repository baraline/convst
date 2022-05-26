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
        

def range_from_x(x, n_steps=8, min_to=None, max_to=None):
    if min_to is None:
        v_min = x.min()*1.075
    else:
        v_min = min_to
        
    if max_to is None:
        v_max = x.max()*1.075
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
        X_mins[i,0] = shapelet_dist_vect(shp, X[i,0]).min()
        X_mins[i,1] = shapelet_dist_vect(shp, X[i,0]).argmin()
        X_mins[i,2] = (shapelet_dist_vect(shp, X[i,0])<threshold).sum()
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
    labels0 = ax0.get_axis_labels(
        x_label=x_labels[0], y_label=y_labels[0]
    ).set_color(label_color).scale(0.6)
    
    
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
    labels1 = ax1.get_axis_labels(
        x_label=x_labels[1], y_label=y_labels[1]
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
    ).scale(0.9).shift(2*DOWN).shift(2.75*RIGHT).scale(0.475)
    x_labels2 = ax2.get_x_axis_label(x_labels[2]).set_color(label_color).scale(0.6)
    y_labels2 = ax2.get_y_axis_label(y_labels[2]).set_color(label_color).scale(0.6).shift(0.5*LEFT)
    
    return VGroup(ax0,labels0), VGroup(ax1,labels1), VGroup(ax2,x_labels2,y_labels2)
    
def get_numerotation(i):
    #Calibrate so it is leftmost to the title at the bar level
    return Tex('{}'.format(i)).scale(0.7)

def make_title(title, i):
    tt = Title(title)
    num = get_numerotation(i).next_to(tt, LEFT)
    return VGroup(tt, num)
    

class Slide(Scene):
    def construct(self):
        x0_index=2
        x1_index=0
        shp0_x_index=np.arange(30,50)
        shp1_x_index=np.arange(102,123)
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
        
        X_Tn_0 = get_features(X, shp0, normalize=True, d=1, threshold=10)
        X_Tn_1 = get_features(X, shp1, normalize=True, d=1, threshold=10)

        ########################################
        #                                      #
        #          INTRODUCE SHAPELETS         #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("What are Shapelets ?", 1)
        vg_all.add(*title)
        
        ax = get_axis_from_x(x0, Tex(''), Tex('')).shift(0.75*DOWN)
        
        graph_x0 = graph_time_series(ax[0], x0, add_vertex_dots=False, line_color=COLOR_0)
        graph_x1 = graph_time_series(ax[0], x1, add_vertex_dots=False, line_color=COLOR_1)
        graph_shp0 = graph_time_series(
            ax[0], shp0_x_index ,y=shp0, add_vertex_dots=False,
            line_color=COLOR_SHP, stroke_width=11
        )

        DataGrp = VGroup(*ax, *graph_x0, *graph_x1)
        vg_all.add(*DataGrp)
        vg_all.add(graph_shp0)
        self.play(FadeIn(title))
        self.wait()
        self.play(Create(ax), run_time=2)
        self.play(Create(graph_x0), run_time=2)
        self.play(Create(graph_x1), run_time=2)
        self.wait(0.5)
        self.play(Create(graph_shp0),run_time=2)
        self.play(DataGrp.animate.stretch(1.8, 0).shift(1.25*UP).scale(0.55))
        self.wait(0.5)
        self.play(graph_shp0.animate.shift(0.25*DOWN).shift(2.5*LEFT))
        # TODO : Add input notation X = ...
        
        Tex_shp = Tex(
            r' A Shapelet $S = \{s_1, \ldots, s_l\}$, with $l$ the length parameter'
        ).scale(0.8).next_to(graph_shp0, RIGHT)
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
        self.play(FadeOut(vg_all))
        
        ########################################
        #                                      #
        #         SHOW ARGMIN Z-Norm           #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Improvments of the base formulation ", 3)
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
                
        _, ax1, ax2 = get_211_axes(x_ranges, y_ranges)
        
        vg_all.add(ax1)
        t = ValueTracker(0)
        x_space = np.arange(norm_dist_vect_shp_0_0.shape[0])
        
        def get_dist_vect():
            i = int(t.get_value())
            return graph_time_series(ax1[0], norm_dist_vect_shp_0_0[:i+1], stroke_color=COLOR_0, add_vertex_dots=False)
                
        def x_subseq_loc():
            i = int(t.get_value())
            l=shp0_norm.shape[0]-1
            x_sub = x0[i:i+shp0_norm.shape[0]]
            vg = VGroup()
            vg.add(Dot(ax0[0].c2p(i, x_sub[0])))
            vg.add(Dot(ax0[0].c2p(i+l, x_sub[l])))
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
            Tex(''), Tex(''), Tex(r"arg$\min d({\cal X},S_1)$")
        ]
        
        y_labels = [
            Tex(''), Tex(''), Tex(r"$\min d({\cal X},S_1)$")
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
        self.play(FadeIn(title))
        self.play(Create(ax0), Create(graph_x0))
        self.play(Create(ax2))
        vg_all.add(*[g_shp_dist_norm1,min0,min1,ax0,graph_x0,all_dots,dots1,dots0,ax2_points,graph_x1])
        
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
        self.play(FadeOut(vg_all))
        self.wait()
        ########################################
        #                                      #
        #     SHOW Contribution summary        #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Our contributions", 4)
        vg_all.add(*title)
        
        
        self.play(FadeIn(title))
        
        
        
        self.wait()
        self.play(FadeOut(vg_all))
        self.wait()
        ########################################
        #                                      #
        #       SHOW Shapelet Occurence        #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Shapelet Occurence", 5)
        vg_all.add(*title)
        #Anim threshold and show effect on points with anim
        # Only two axes 
        self.play(FadeIn(title))
        
        
        
        self.wait()
        self.play(FadeOut(vg_all))
        self.wait()
        ########################################
        #                                      #
        #        SHOW Dilated Shapelets        #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Dilated Shapelet", 6)
        vg_all.add(*title)
        
        self.play(FadeIn(title))
        
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
        
        self.wait()
        self.play(FadeOut(vg_all))
        self.wait()
        ########################################
        #                                      #
        #              SHOW RDST               #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Random Dilated Shapelet Transform", 7)
        vg_all.add(*title)
        self.play(FadeIn(title))
        
        
        
        self.wait()
        self.play(FadeOut(vg_all))
        self.wait()
        ########################################
        #                                      #
        #           TIMING Results             #
        #                                      # 
        ########################################
        vg_all = VGroup()
        title = make_title("Time Complexity", 8)
        vg_all.add(*title)
        self.play(FadeIn(title))
        
        
        
        self.wait()
        self.play(FadeOut(vg_all))
        self.wait()
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

    """
      
        
        ###############################################################
        
       
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