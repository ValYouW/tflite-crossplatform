#import <Foundation/Foundation.h>
#import <CoreMedia/CoreMedia.h>

NS_ASSUME_NONNULL_BEGIN

@interface CV : NSObject
- (void) initDetector;
;
- (NSArray*) detect: (CMSampleBufferRef)buffer;
@end

NS_ASSUME_NONNULL_END
