import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Text("Metal Neural Network Example")
                .padding()
            Button("Run Neural Network") {
                let viewController = ViewController()
                viewController.viewDidLoad()
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
