scalaVersion := "2.11.8"

name := "stat"

organization := "io.github.pityka"

version := "0.0.1"

libraryDependencies ++= Seq(
  "io.github.pityka" %% "saddle-linalg" % "0.0.6",
  "org.scalatest" %% "scalatest" % "3.0.0" % "test"
)

reformatOnCompileSettings

pomExtra in Global := {
  <url>https://pityka.github.io/stat</url>
  <licenses>
    <license>
      <name>MIT</name>
      <url>https://opensource.org/licenses/MIT</url>
    </license>
  </licenses>
  <scm>
    <connection>scm:git:github.com/pityka/stat</connection>
    <developerConnection>scm:git:git@github.com:pityka/stat</developerConnection>
    <url>github.com/pityka/stat</url>
  </scm>
  <developers>
    <developer>
      <id>pityka</id>
      <name>Istvan Bartha</name>
      <url>https://pityka.github.io/stat/</url>
    </developer>
  </developers>
}
