from keras.layers import *#Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.models import Model

def FCN32(n_classes):
#VGG
    img_input = Input(shape=(512, 512, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
 
    #L2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #L3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #L4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #L5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
#FCN32
    vgg=Model(img_input, x)
    #vgg.summary()
    vgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5", by_name=True)
    for layer in vgg.layers:
        layer.trainable = False
    o = Conv2D(4096, (7, 7), activation='relu', padding='same')(vgg.output)
    
    #o = Conv2D(4096, (7, 7), activation='relu', padding='same')(x)
    o = Dropout(0.5)(o)
    o = Conv2D(4096, (1, 1), activation='relu', padding='same')(o)
    o = Dropout(0.5)(o)
    
    
    o = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal'))(o)
    o = Conv2DTranspose(n_classes , kernel_size=(64,64) ,  strides=(32,32), \
        padding='same' , use_bias=False)(o)
    #o_shape = Model(img_input , o ).output_shape
    
    #print("koko" , o_shape)
    
    #o = (Reshape(( -1  , outputHeight*outputWidth)))(o)
    #o = (Permute((2, 3, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5", by_name=True)
    #model.summary()
    return model

def crop( o1 , o2 , i  ):
	o_shape2 = Model( i  , o2 ).output_shape
	outputHeight2 = o_shape2[2]
	outputWidth2 = o_shape2[3]

	o_shape1 = Model( i  , o1 ).output_shape
	outputHeight1 = o_shape1[2]
	outputWidth1 = o_shape1[3]

	cx = abs( outputWidth1 - outputWidth2 )
	cy = abs( outputHeight2 - outputHeight1 )

	if outputWidth1 > outputWidth2:
		o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )) )(o1)
	else:
		o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )) )(o2)
	
	if outputHeight1 > outputHeight2 :
		o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )) )(o1)
	else:
		o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )) )(o2)

	return o1 , o2 
    
def FCN8(n_classes):
#VGG
    img_input = Input(shape=(512, 512, 3))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
 
    #L2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #L3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x
    
    #L4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x
    
    #L5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x
    
#FCN8
    vgg=Model(img_input, x)
    #vgg.summary()
    vgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5", by_name=True)
    for layer in vgg.layers:
       layer.trainable = False
   
    o = f5
    
    o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same' ))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same' ))(o)
    o = Dropout(0.5)(o)
    
    o = ( Conv2D( n_classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal'  ))(o)
    o = Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False  )(o)
    
    o2 = f4
    o2 = ( Conv2D( n_classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal'  ))(o2)
    
    o , o2 = crop( o , o2 , img_input )
    
    o = Add()([ o , o2 ])
    
    o = Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False  )(o)
    o2 = f3 
    o2 = ( Conv2D( n_classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal'  ))(o2)
    o2 , o = crop( o2 , o , img_input )
    o  = Add()([ o2 , o ])
    
    
    o = Conv2DTranspose( n_classes , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False  )(o)
    
    
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.summary()
    
    return model
    
def VGGUnet( n_classes ,  input_height=512, input_width=512 , vgg_level=3):
    
    img_input = Input(shape=(input_height,input_width,3))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'  )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'  )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'  )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'  )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'  )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'  )(x)
    f2 = x
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'  )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'  )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'  )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'  )(x)
    f3 = x
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'  )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'  )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'  )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'  )(x)
    f4 = x
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'  )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'  )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'  )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'  )(x)
    f5 = x
    vgg  = Model(  img_input , x  )
    vgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5", by_name=True)
    
    
    levels = [f1 , f2 , f3 , f4 , f5 ]
    
    o = f4
    
    o = ( ZeroPadding2D( (1,1)))(o)
    o = ( Conv2D(512, (3, 3), padding='valid' ))(o)
    o = ( BatchNormalization())(o)
    
    o = (UpSampling2D( (2,2) ))(o)
    o = ( concatenate([ o ,f3],axis=3 )  )
    o = ( ZeroPadding2D( (1,1) ))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid' ))(o)
    o = ( BatchNormalization())(o)
    
    o = (UpSampling2D( (2,2) ))(o)
    o = ( concatenate([o,f2],axis=3 ) )
    o = ( ZeroPadding2D((1,1)   ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid'   ) )(o)
    o = ( BatchNormalization())(o)
    
    o = (UpSampling2D( (2,2) ))(o)
    o = ( concatenate([o,f1],axis=3 ) )
    o = ( ZeroPadding2D((1,1)    ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'    ))(o)
    o = ( BatchNormalization())(o)
    o = (UpSampling2D( (2,2) ))(o)
    
    o =  Conv2D( n_classes , (3, 3) , padding='same'  )( o )
    o_shape = Model(img_input , o ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]
    
    #o = (Reshape((  n_classes , outputHeight*outputWidth   )))(o)
    #o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    #model.outputWidth = outputWidth
    #model.outputHeight = outputHeight
    model.summary()
    
    return model