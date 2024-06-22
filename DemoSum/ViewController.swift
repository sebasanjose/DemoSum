import Cocoa
import Metal
import MetalKit

class ViewController: NSViewController {
    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var pipelineState: MTLComputePipelineState!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Initialize Metal
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()
        
        // Load the shader
        let defaultLibrary = device.makeDefaultLibrary()
        let kernelFunction = defaultLibrary?.makeFunction(name: "forward_pass")
        do {
            pipelineState = try device.makeComputePipelineState(function: kernelFunction!)
        } catch {
            fatalError("Unable to create pipeline state")
        }
        
        // Prepare data
        let inputLength = 10000000 // Large input length for performance demonstration
        let featureCount = 2
        var inputs: [Float] = []
        for _ in 0..<inputLength {
            let randomFeature1 = Float.random(in: 0...1)
            let randomFeature2 = Float.random(in: 0...1)
            inputs.append(contentsOf: [randomFeature1, randomFeature2])
        }
        
        let weights: [Float] = [0.5, -0.5] // Example weights
        let biases: [Float] = [0.1] // Example bias
        
        var outputsGPU = [Float](repeating: 0, count: inputLength)
        var outputsCPU = [Float](repeating: 0, count: inputLength)
        
        // Run on GPU
        let gpuStartTime = CFAbsoluteTimeGetCurrent()
        runOnGPU(inputs: inputs, weights: weights, biases: biases, outputs: &outputsGPU, inputLength: inputLength, featureCount: featureCount)
        let gpuEndTime = CFAbsoluteTimeGetCurrent()
        
        // Run on CPU
        let cpuStartTime = CFAbsoluteTimeGetCurrent()
        runOnCPU(inputs: inputs, weights: weights, biases: biases, outputs: &outputsCPU, inputLength: inputLength, featureCount: featureCount)
        let cpuEndTime = CFAbsoluteTimeGetCurrent()
        

        
//        // Check a subset of results
//        for i in 0..<1000000 {
//            print("GPU Output[\(i)] = \(outputsGPU[i]), CPU Output[\(i)] = \(outputsCPU[i])")
//        }
        
        // Print execution times
        print("GPU Execution Time: \(gpuEndTime - gpuStartTime) seconds")
        print("CPU Execution Time: \(cpuEndTime - cpuStartTime) seconds")
    }
    
    func runOnGPU(inputs: [Float], weights: [Float], biases: [Float], outputs: inout [Float], inputLength: Int, featureCount: Int) {
        let inputsBuffer = device.makeBuffer(bytes: inputs, length: inputLength * featureCount * MemoryLayout<Float>.size, options: [])
        let weightsBuffer = device.makeBuffer(bytes: weights, length: featureCount * MemoryLayout<Float>.size, options: [])
        let biasesBuffer = device.makeBuffer(bytes: biases, length: MemoryLayout<Float>.size, options: [])
        let outputsBuffer = device.makeBuffer(length: inputLength * MemoryLayout<Float>.size, options: [])
        
        // Create command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()
        let computeEncoder = commandBuffer?.makeComputeCommandEncoder()
        computeEncoder?.setComputePipelineState(pipelineState)
        computeEncoder?.setBuffer(inputsBuffer, offset: 0, index: 0)
        computeEncoder?.setBuffer(weightsBuffer, offset: 0, index: 1)
        computeEncoder?.setBuffer(biasesBuffer, offset: 0, index: 2)
        computeEncoder?.setBuffer(outputsBuffer, offset: 0, index: 3)
        
        // Dispatch compute commands
        let gridSize = MTLSize(width: inputLength, height: 1, depth: 1)
        var threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup
        if threadGroupSize > inputLength {
            threadGroupSize = inputLength
        }
        let threadgroupSize = MTLSize(width: threadGroupSize, height: 1, depth: 1)
        computeEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        computeEncoder?.endEncoding()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        
        // Retrieve results
        let outputsPointer = outputsBuffer?.contents().bindMemory(to: Float.self, capacity: inputLength)
        for i in 0..<inputLength {
            outputs[i] = outputsPointer![i]
        }
    }
    
    func runOnCPU(inputs: [Float], weights: [Float], biases: [Float], outputs: inout [Float], inputLength: Int, featureCount: Int) {
        for i in 0..<inputLength {
            var dot_product = 0.0
            for j in 0..<featureCount {
                dot_product += Double(inputs[i * featureCount + j] * weights[j])
            }
            dot_product += Double(biases[0])
            outputs[i] = Float(1.0 / (1.0 + exp(-dot_product)))
        }
    }
}
