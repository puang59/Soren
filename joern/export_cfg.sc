// Export the control flow graph of every method in a CPG as JSON lines.
//
//   joern --script joern/export_cfg.sc --param cpgFile=<batch>.cpg.bin --param outFile=<batch>.jsonl
//
// One JSON object per method:
//   file      source file name, which carries the sample id
//   name      method name
//   line      first line of the method; line_end: its last line
//   nodes     [id, label, line, name, code] for METHOD, METHOD_RETURN and every node on a CFG
//             edge. `name` is the callee or operator name for CALL nodes, else "".
//   edges     [source id, target id] CFG edges
//   controls  [type, line, header_first_line, header_last_line] per control structure. The
//             header is the condition (plus init and update for FOR); it is empty for
//             BREAK, CONTINUE, GOTO and ELSE, where both header lines equal `line`.
//   stmts     [label, first_line, last_line] for every statement directly inside a block or
//             forming the unbraced body of a control structure, spanning all lines of its
//             expression tree. Control structures are left out; their extent is the header.

import io.shiftleft.codepropertygraph.generated.nodes.*
import io.shiftleft.semanticcpg.language.*
import java.io.{BufferedWriter, FileWriter}

def quote(text: String): String = {
  val out = new StringBuilder("\"")
  text.foreach {
    case '"'           => out.append("\\\"")
    case '\\'          => out.append("\\\\")
    case '\n'          => out.append("\\n")
    case '\r'          => out.append("\\r")
    case '\t'          => out.append("\\t")
    case c if c < ' '  => out.append(f"\\u${c.toInt}%04x")
    case c             => out.append(c)
  }
  out.append("\"").toString
}

def line(node: StoredNode): Int = node match {
  case n: AstNode => n.lineNumber.getOrElse(-1)
  case _          => -1
}

def span(nodes: Iterator[AstNode]): Option[(Int, Int)] = {
  val lines = nodes.flatMap(_.ast.lineNumber).toList
  if (lines.isEmpty) None else Some((lines.min, lines.max))
}

@main def exec(cpgFile: String, outFile: String) = {
  importCpg(cpgFile)
  val writer = new BufferedWriter(new FileWriter(outFile))
  var count = 0
  cpg.method.filterNot(_.isExternal).filterNot(_.name.startsWith("<")).foreach { method =>
    val inner = method.cfgNode.l
    val all: List[CfgNode] = method :: method.methodReturn :: inner
    val edges = all.flatMap(n => n._cfgOut.map(t => (n.id, t.id)))
    val onEdge = edges.flatMap((a, b) => List(a, b)).toSet + method.id + method.methodReturn.id

    val nodes = all.filter(n => onEdge.contains(n.id)).map { n =>
      val name = n match {
        case call: Call => call.name
        case _          => ""
      }
      s"[${n.id},${quote(n.label)},${line(n)},${quote(name)},${quote(n.code.take(200))}]"
    }

    val controls = method.controlStructure.l.map { cs =>
      val own = cs.lineNumber.getOrElse(-1)
      val header: Iterator[AstNode] =
        if (cs.controlStructureType == "FOR") cs.astChildren.l.dropRight(1).iterator
        else cs.condition
      val (first, last) = span(header).getOrElse((own, own))
      s"[${quote(cs.controlStructureType)},$own,$first,$last]"
    }

    // Statements sit directly inside a block, or are the unbraced body of a control structure.
    val unbraced = method.controlStructure.l.flatMap { cs =>
      val header = if (cs.controlStructureType == "FOR") cs.astChildren.l.dropRight(1) else cs.condition.l
      cs.astChildren.l.filterNot(c => header.exists(_.id == c.id))
    }
    val stmts = (method.ast.isBlock.astChildren.l ++ unbraced)
      .filterNot(c => c.isInstanceOf[ControlStructure] || c.isInstanceOf[Block])
      .distinctBy(_.id)
      .flatMap(s => span(Iterator(s)).map((first, last) => s"[${quote(s.label)},$first,$last]"))

    val fields = List(
      s"\"file\":${quote(method.filename)}",
      s"\"name\":${quote(method.name)}",
      s"\"line\":${method.lineNumber.getOrElse(-1)}",
      s"\"line_end\":${method.lineNumberEnd.getOrElse(-1)}",
      s"\"nodes\":[${nodes.mkString(",")}]",
      s"\"edges\":[${edges.map((a, b) => s"[$a,$b]").mkString(",")}]",
      s"\"controls\":[${controls.mkString(",")}]",
      s"\"stmts\":[${stmts.mkString(",")}]"
    )
    writer.write(fields.mkString("{", ",", "}"))
    writer.newLine()
    count += 1
  }
  writer.close()
  println(s"exported $count methods to $outFile")
}
