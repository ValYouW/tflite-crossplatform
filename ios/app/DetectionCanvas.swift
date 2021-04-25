import Foundation
import UIKit

// Used to draw detection rectangles on screen
class DetectionsCanvas: UIView {
    var labelmap = [String]()
    var detections = [Float]() // Raw results from detector

    // The size of the image we run detection on
    var capFrameWidth = 0
    var capFrameHeight = 0
    
    override func draw(_ rect: CGRect) {
        if (detections.count < 1) {return}
        if (detections.count % 6 > 0) {return;} // Each detection should have 6 numbers (classId, scrore, xmin, xmax, ymin, ymax)

        guard let context = UIGraphicsGetCurrentContext() else {return}
        context.clear(self.frame)

        // detection coords are in frame coord system, convert to screen coords
        let scaleX = self.frame.size.width / CGFloat(capFrameWidth)
        let scaleY = self.frame.size.height / CGFloat(capFrameHeight)

        // The camera view offset on screen
        let xoff = self.frame.minX
        let yoff = self.frame.minY
        
        let count = detections.count / 6
        for i in 0..<count {
            let idx = i * 6
            let classId = Int(detections[idx])
            let score = detections[idx + 1]
            if (score < 0.6) {continue}
            
            let xmin = xoff + CGFloat(detections[idx + 2]) * scaleX
            let xmax = xoff + CGFloat(detections[idx + 3]) * scaleX
            let ymin = yoff + CGFloat(detections[idx + 4]) * scaleY
            let ymax = yoff + CGFloat(detections[idx + 5]) * scaleY
            
            let labelIdx = classId
            let label = labelmap.count > labelIdx ? labelmap[labelIdx] : classId.description

            // Draw rect
            context.beginPath()
            context.move(to: CGPoint(x: xmin, y: ymin))
            context.addLine(to: CGPoint(x: xmax, y: ymin))
            context.addLine(to: CGPoint(x: xmax, y: ymax))
            context.addLine(to: CGPoint(x: xmin, y: ymax))
            context.addLine(to: CGPoint(x: xmin, y: ymin))

            context.setLineWidth(2.0)
            context.setStrokeColor(UIColor.red.cgColor)
            context.drawPath(using: .stroke)

            // Draw label
            UIGraphicsPushContext(context)
            let font = UIFont.systemFont(ofSize: 30)
            let string = NSAttributedString(string: label, attributes: [NSAttributedString.Key.font: font, NSAttributedString.Key.foregroundColor: UIColor.red])
            string.draw(at: CGPoint(x: xmin, y: ymin))
        }
        
        UIGraphicsPopContext()
    }
}
