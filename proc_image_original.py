def imread_x(f,path,box=(0,792,0,85)):
    return color.rgb2gray(io.imread(
            r'D:\Mestrado\Imagens\Antigas'
            + path + '(%d).jpg'%f)[box[0]:box[1],
                                   box[2]:box[3]])

def im_start(pic,path,box):
    im_matrix = io.ImageCollection(
            pic,path=path,box=box,
            load_func=imread_x)
    return(im_matrix.concatenate())

def bg_im(im):
    bg = np.zeros(im.shape,dtype=int)
    bg[:,0:np.floor_divide(im.shape[1],2)] = 255
    return(bg)

def bg_removal(im_mat):
    bg = bg_im(im_mat[0])
    im_int = im_mat * 255
    im_int = im_int.astype(int)
    no_bg = im_int - bg
    no_bg[no_bg < 0] = 0
    return(no_bg)

def im_proc(im):
    th = threshold_otsu(im)
    im_bin = im > th
    return(ndi.binary_fill_holes(
                morphology.closing(
                im_bin,np.ones((3,3)))))

def interface_height(im,scale):
    D = {n: np.array([i for i, j in enumerate(k) if j]) for n, k in enumerate(im)}
    height = []#np.array([np.amin(i) for i in list(D.values()) if list(D.values())])
    for v in D.values():
        if len(v) == 0:
            height.append(height[-1])
        else:
            height.append(np.min(v))
    return(D,height)