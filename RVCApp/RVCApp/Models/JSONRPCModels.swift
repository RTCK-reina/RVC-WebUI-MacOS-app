import Foundation

// MARK: - JSON-RPC 2.0 primitives

/// Requests sent from Swift to the Python backend.
struct JSONRPCRequest<P: Encodable>: Encodable {
    let jsonrpc: String = "2.0"
    let id: Int
    let method: String
    let params: P
}

/// Minimal request used for call sites that don't need typed params — we
/// carry the raw JSON value instead.
struct JSONRPCRawRequest: Encodable {
    let jsonrpc: String = "2.0"
    let id: Int
    let method: String
    let params: JSONValue
}

/// Responses / notifications flowing from Python to Swift.
struct JSONRPCEnvelope: Decodable {
    let jsonrpc: String?
    let id: Int?
    let method: String?
    let result: JSONValue?
    let error: JSONRPCError?
    let params: JSONValue?
}

struct JSONRPCError: Decodable, Error {
    let code: Int
    let message: String
    let data: JSONValue?
}

// MARK: - Type-erased JSON value

/// Flexible JSON value used when we don't want to commit to a strict schema
/// (progress payloads, resource stats, method params from notifications).
indirect enum JSONValue: Codable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
            return
        }
        if let b = try? c.decode(Bool.self) {
            self = .bool(b); return
        }
        if let d = try? c.decode(Double.self) {
            self = .number(d); return
        }
        if let s = try? c.decode(String.self) {
            self = .string(s); return
        }
        if let a = try? c.decode([JSONValue].self) {
            self = .array(a); return
        }
        if let o = try? c.decode([String: JSONValue].self) {
            self = .object(o); return
        }
        throw DecodingError.dataCorruptedError(
            in: c,
            debugDescription: "Unknown JSON value")
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    // MARK: Convenience accessors

    var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    var doubleValue: Double? {
        if case .number(let d) = self { return d }
        if case .string(let s) = self { return Double(s) }
        if case .bool(let b) = self { return b ? 1 : 0 }
        return nil
    }

    var intValue: Int? {
        if let d = doubleValue { return Int(d) }
        return nil
    }

    var boolValue: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }

    var arrayValue: [JSONValue]? {
        if case .array(let a) = self { return a }
        return nil
    }

    var objectValue: [String: JSONValue]? {
        if case .object(let o) = self { return o }
        return nil
    }

    subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }
}
