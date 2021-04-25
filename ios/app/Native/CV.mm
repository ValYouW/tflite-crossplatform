#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>
#import <opencv2/imgcodecs/ios.h>
#import "CV.h"
#import "ObjectDetector.h"

using namespace std;
using namespace cv;

static ObjectDetector* detector = nil;

@implementation CV

- (void) initDetector {
    if(detector != nil) {
        return;
    }

    // Load the graph config resource.
    long size = 0;
    char* model = nullptr;
    NSError* configLoadError = nil;
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"od_model" ofType:@"tflite"];
    NSData* data = [NSData dataWithContentsOfFile:modelPath options:0 error:&configLoadError];
    if (!data) {
      NSLog(@"Failed to load model: %@", configLoadError);
    } else {
        size = data.length;
        model = (char*)data.bytes;
    }

    detector = new ObjectDetector((const char*)model, size, false);
}

-(NSArray*) detect: (CMSampleBufferRef)buffer {
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(buffer);
    CVPixelBufferLockBaseAddress( pixelBuffer, 0 );

    //Processing here
    int bufferWidth = (int)CVPixelBufferGetWidth(pixelBuffer);
    int bufferHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    unsigned char *pixel = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);

    //put buffer in open cv, no memory copied
    Mat dst = Mat(bufferHeight, bufferWidth, CV_8UC4, pixel, CVPixelBufferGetBytesPerRow(pixelBuffer));

    //End processing
    CVPixelBufferUnlockBaseAddress( pixelBuffer, 0 );

    [self initDetector];
    
    // Run detections
    DetectResult* detections = detector->detect(dst);

    // decode detections into float array
    NSMutableArray *array = [[NSMutableArray alloc] initWithCapacity: (detector->DETECT_NUM * 6)];

    for (int i = 0; i < detector->DETECT_NUM; ++i) {
        [array addObject:[NSNumber numberWithFloat:detections[i].label]];
        [array addObject:[NSNumber numberWithFloat:detections[i].score]];
        [array addObject:[NSNumber numberWithFloat:detections[i].xmin]];
        [array addObject:[NSNumber numberWithFloat:detections[i].xmax]];
        [array addObject:[NSNumber numberWithFloat:detections[i].ymin]];
        [array addObject:[NSNumber numberWithFloat:detections[i].ymax]];
    }
    
    return array;
}

@end
