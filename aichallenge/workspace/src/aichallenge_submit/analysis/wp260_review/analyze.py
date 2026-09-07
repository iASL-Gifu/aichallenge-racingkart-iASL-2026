"""Static CSV/lane-center inspection; not a replay of live MPC or ROS boundaries."""
from pathlib import Path
import ast, math, json
from types import SimpleNamespace as NS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'multi_purpose_mpc_ros'
t=np.genfromtxt(root/'env/centerline/traj_center_mincurv_capped.csv',delimiter=',',names=True)
b=np.genfromtxt(root/'env/centerline/waypoint_bounds_center_mincurv_capped.csv',delimiter=',',names=True)
p=np.column_stack([t['x_m'],t['y_m']]);vec=np.roll(p,-1,axis=0)-p;yaw=np.arctan2(vec[:,1],vec[:,0]);normal=np.column_stack([-np.sin(yaw),np.cos(yaw)])
source=(root/'multi_purpose_mpc_ros/core/reference_path.py').read_text();tree=ast.parse(source)
margin=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='OUTER_COURSE_MARGIN' for k in n.targets))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReferencePath');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_lane_bounds');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<real-lane-bounds>','exec'),ns)
ref=NS(n_lanes=3,inner_lane_width=.5,get_waypoint=lambda i:NS(ub=b['ub'][i]+0-margin,lb=b['lb'][i]+margin))
lanes=np.array([ns['get_lane_bounds'](ref,i) for i in range(len(p))]);centers=lanes.mean(axis=2)
paths={'Center':p,**{f'L{i}':p+normal*centers[:,i,None] for i in (0,1,2)}}
def curvature(q):
 a=q-np.roll(q,1,axis=0);c=np.roll(q,-1,axis=0)-q
 return 2*(a[:,0]*c[:,1]-a[:,1]*c[:,0])/(np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1)*np.linalg.norm(a+c,axis=1))
kmax=math.tan(math.radians(18))/1.087
stats={};window=np.arange(230,281)
for name,q in paths.items():
 k=curvature(q);delta=np.arctan(1.087*k);ds=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
 rate=np.abs(np.roll(delta,-1)-delta)/ds
 i=window[np.argmax(abs(k[window]))]
 stats[name]={'max_abs_curvature':float(abs(k[i])),'wp':int(i),'required_steer_deg':float(abs(np.degrees(delta[i]))),'max_steer_rate_at_7_5_mps':float(max(rate[window])*7.5),'exceeding_18deg_wps':window[abs(k[window])>kmax].tolist()}
print(json.dumps({'margin':margin,'kappa_limit':kmax,'radius_limit':1/kmax,'stats':stats},indent=2))
Path(__file__).with_name('metrics.json').write_text(json.dumps({'margin':margin,'kappa_limit':kmax,'stats':stats},indent=2))
left=p+normal*b['ub'][:,None];right=p+normal*b['lb'][:,None]
fig,axs=plt.subplots(1,2,figsize=(13,6));origin=p[260]
for name,q in paths.items():axs[0].plot(*(q[window]-origin).T,label=name,linewidth=1.3)
for name,q in [('CSV left',left),('CSV right',right)]:axs[0].plot(*(q[window]-origin).T,'k-',linewidth=2,label=name)
for i in range(230,281,5):axs[0].annotate(str(i),p[i]-origin,fontsize=8)
axs[0].axis('equal');axs[0].legend();axs[0].set(xlabel='map x relative to WP260 [m]',ylabel='map y [m]',title='Static bounds and nominal lane centers')
for name,q in paths.items():axs[1].plot(window,curvature(q)[window],label=name)
axs[1].axhline(kmax,color='red',linestyle='--',label='18 deg bicycle limit');axs[1].axhline(-kmax,color='red',linestyle='--');axs[1].axvspan(230,255,alpha=.08,color='red',label='L0 entry prohibition');axs[1].set(xlabel='Center WP',ylabel='three-point curvature [1/m]',title='Geometric curvature (not solved MPC)');axs[1].legend();axs[1].grid();fig.tight_layout();fig.savefig(Path(__file__).with_name('geometry.png'),dpi=150)
print('WP, lb, ub, L0 center, Center kappa, L0 kappa')
for i in range(250,271):print(i,round(b['lb'][i],3),round(b['ub'][i],3),round(centers[i,0],3),round(curvature(p)[i],3),round(curvature(paths['L0'])[i],3))
