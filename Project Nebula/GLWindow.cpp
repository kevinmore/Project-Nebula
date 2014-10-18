#include <gl/glew.h>
#include <GLWindow.h>
#include <Qt/qapplication.h>
#include <Qt/qdesktopwidget.h>
#include <Qt/qdebug.h>
#include <Utility/MeshGenerator.h>
#include <Utility/MeshImporter.h>


GLWindow::GLWindow(QWidget *parent) : QGLWidget(QGLFormat(/* Additional format options */), parent)
{
//	setFormat(QGLFormat(QGL::Rgba | QGL::DoubleBuffer | QGL::DepthBuffer));
//	m_allRenderableObjects.reserve(1);
	m_frameCount = 0;


	lightAngle = 0;
	handRotationAngle = 1;
	handRotationDirection = 1;

	m_timer = new QTimer(this);
	connect(m_timer, SIGNAL(timeout()), this, SLOT(updateLoop()));
	m_timer->start(0);
}

QSize GLWindow::sizeHint() const
{
	return QSize(1024, 768);
}

GLWindow::~GLWindow(void)
{
	hand->freeSkeleton(hand);
//	glUseProgram(0);
// 	Shader* temp = 0;
// 	for (unsigned int i = 0; i < m_runningShaderPrograms.size(); ++i)
// 	{
// 		temp = m_runningShaderPrograms[i];
// 		temp->DelteBuffers();
// 		glDeleteProgram(temp->ShaderProgramID());
// 	}
}

void GLWindow::initializeGL()
{

#ifdef WIN32
	glActiveTexture = (PFNGLACTIVETEXTUREPROC) wglGetProcAddress((LPCSTR) "glActiveTexture");
#endif

	glShadeModel(GL_SMOOTH);
	glClearDepth( 1.0 );

	// enable the depth buffer
	glEnable(GL_DEPTH_TEST);
	//glDepthFunc( GL_LEQUAL );

	//glEnable(GL_CULL_FACE);

	qglClearColor(QColor(40, 40, 40));

	
	// spot light

	coloringShaderProgram.addShaderFromSourceFile(QGLShader::Vertex, "../Resource/Shaders/coloringVertexShader.vsh");
	coloringShaderProgram.addShaderFromSourceFile(QGLShader::Fragment, "../Resource/Shaders/coloringFragmentShader.fsh");
	coloringShaderProgram.link();

	QVector<QVector3D> spotlightVertices;
	QVector<QVector3D> spotlightColors;

	spotlightVertices << QVector3D(   0,    0.1,    0) << QVector3D(-0.05,    0,  0.05) << QVector3D( 0.05,    0,  0.05) // Front
						<< QVector3D(   0,    0.1,    0) << QVector3D( 0.05,    0, -0.05) << QVector3D(-0.05,    0, -0.05) // Back
						<< QVector3D(   0,    0.1,    0) << QVector3D(-0.05,    0, -0.05) << QVector3D(-0.05,    0,  0.05) // Left
						<< QVector3D(   0,    0.1,    0) << QVector3D( 0.05,    0,  0.05) << QVector3D( 0.05,    0, -0.05) // Right
						<< QVector3D(-0.05,    0, -0.05) << QVector3D( 0.05,    0, -0.05) << QVector3D( 0.05,    0,  0.05) // Bottom
						<< QVector3D( 0.05,    0,  0.05) << QVector3D(-0.05,    0,  0.05) << QVector3D(-0.05,    0, -0.05);
	spotlightColors << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) // Front
					<< QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) // Back
					<< QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) // Left
					<< QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) // Right
					<< QVector3D(  0.9,   0.9,   0.9) << QVector3D(  0.9,   0.9,   0.9) << QVector3D(  0.9,   0.9,   0.9) // Bottom
					<< QVector3D(  0.9,   0.9,   0.9) << QVector3D(  0.9,   0.9,   0.9) << QVector3D(  0.9,   0.9,   0.9);
	
	numSpotlightVertices = 18;
	spotlightBuffer.create();
	spotlightBuffer.bind();
	spotlightBuffer.setUsagePattern(QGLBuffer::DynamicDraw);
	spotlightBuffer.allocate(numSpotlightVertices * (3 + 3) * sizeof(GLfloat));

	int offset = 0;
	spotlightBuffer.write(offset, spotlightVertices.constData(), numSpotlightVertices * 3 * sizeof(GLfloat));
	offset += numSpotlightVertices * 3 * sizeof(GLfloat);
	spotlightBuffer.write(offset, spotlightColors.constData(), numSpotlightVertices * 3 * sizeof(GLfloat));

	spotlightBuffer.release();
	/***********************************************************************************************************/

	// cylinder
	lightingShaderProgram.addShaderFromSourceFile(QGLShader::Vertex, "../Resource/Shaders/lightingVertexShader.vsh");
	lightingShaderProgram.addShaderFromSourceFile(QGLShader::Fragment, "../Resource/Shaders/lightingFragmentShader.fsh");
	lightingShaderProgram.link();

	//shape = MeshGenerator::makeCylinder(0.05f, 0.025f, 0.020f, vec4(0.855f, 0.745f, 0.702f, 1.0f), vec4(0.847f, 0.808f, 0.8f, 1.0f), 256);
	shape2 = MeshGenerator::makeCylinder(0.5f, 0.3f, 0.1f, vec4(0.455f, 0.245f, 0.702f, 1.0f), vec4(0.847f, 0.808f, 0.2f, 1.0f), 256);

	// make a hand
	hand = MeshGenerator::makeHand();

	// import the hand mesh
	MeshImporter importer;
	if (importer.loadMeshFromFile("../Resource/Models/phoenix_ugv.md2"))
	{
		Bone* root = importer.getSkeleton();
	}
	int i = 1;
}

void GLWindow::resizeGL( int w, int h )
{
	if ( h == 0 ) h = 1;

	pMatrix.setToIdentity();
	pMatrix.perspective(60.0, (float) w / (float) h, 0.001, 1000);
	// set the view port the the full qt window
	glViewport( 0, 0, w, h );

}

void GLWindow::paintGL()
{
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	
	/*********************set up the world to view matrix****************************************/
	vMatrix.setToIdentity();
	vMatrix.lookAt(m_camera.Position(), m_camera.ViewDirection() + m_camera.Position(), m_camera.Up());

	/*********************set up the spinning spot light****************************************/
	mMatrix.setToIdentity();

	QMatrix4x4 mvMatrix;
	mvMatrix = vMatrix * mMatrix;

	normalMatrix = mvMatrix.normalMatrix();

	QMatrix4x4 lightTransformation;
	lightTransformation.rotate(lightAngle, 0, 1, 0);

	lightPosition = lightTransformation * QVector3D(0, 0.14, 0.1);
	mMatrix.setToIdentity();
	mMatrix.translate(lightPosition);
	mMatrix.rotate(lightAngle, 0, 1, 0);
	mMatrix.rotate(45, 1, 0, 0);
	mMatrix.scale(0.1);

	coloringShaderProgram.bind();

	coloringShaderProgram.setUniformValue("mvpMatrix", pMatrix * vMatrix * mMatrix);

	spotlightBuffer.bind();
	int offset = 0;
	coloringShaderProgram.setAttributeBuffer("vertex", GL_FLOAT, offset, 3, 0);
	coloringShaderProgram.enableAttributeArray("vertex");
	offset += numSpotlightVertices * 3 * sizeof(GLfloat);
	coloringShaderProgram.setAttributeBuffer("color", GL_FLOAT, offset, 3, 0);
	coloringShaderProgram.enableAttributeArray("color");
	spotlightBuffer.release();

	glDrawArrays(GL_TRIANGLES, 0, numSpotlightVertices);

	coloringShaderProgram.disableAttributeArray("vertex");
	coloringShaderProgram.disableAttributeArray("color");
	coloringShaderProgram.release();

	/*********************Render other objects*************************************/
	
	// change the animation direction every 70 frames
	++m_frameCount;
	if (m_frameCount % 70 == 0)
	{
		handRotationAngle *= -1;
	}

	mat4 transform;
	// rotate the palm around the Y axis, for demo purpose
//  	transform.rotate(0.2, vec3(0, 1, 0));
//  	hand->m_localTransform *= transform;
	Bone::sortSkeleton(hand);

	// only move the 5 fingers, not the palm
	transform.setToIdentity();
	transform.rotate(-handRotationAngle, vec3(1, 0, 0));
	for (int i = 0; i < hand->childCount(); ++i)
	{
		Bone::configureSkeleton(hand->getChild(i), transform);
	}
	

	renderSkeleton(hand);
}

void GLWindow::renderMesh( QGLShaderProgram &shader, MeshData &mesh, mat4 &modelToWorldMatrix )
{
	// calculate MV Matrix
	mat4 mvMatrix = vMatrix * modelToWorldMatrix;

	// active the shader
	shader.bind();

	// set the uniform values
	shader.setUniformValue("mvpMatrix", pMatrix * mvMatrix);
	shader.setUniformValue("mvMatrix", mvMatrix);
	shader.setUniformValue("normalMatrix", normalMatrix);
	shader.setUniformValue("lightPosition", vMatrix * lightPosition);
	shader.setUniformValue("ambientColor", QColor(32, 32, 32));
	shader.setUniformValue("diffuseColor", QColor(128, 128, 128));
	shader.setUniformValue("specularColor", QColor(255, 255, 255));
	shader.setUniformValue("ambientReflection", (GLfloat) 1.0);
	shader.setUniformValue("diffuseReflection", (GLfloat) 1.0);
	shader.setUniformValue("specularReflection", (GLfloat) 1.0);
	shader.setUniformValue("shininess", (GLfloat) 100.0);
	shader.setUniformValue("texture", 0);

	// active the buffer from the mesh itself 
	// pass in values
	// lastly, release the buffer
	mesh.vertexBuff.bind();
	shader.setAttributeBuffer("vertex", GL_FLOAT, 0, 3, sizeof(Vertex));
	shader.enableAttributeArray("vertex");
	shader.setAttributeBuffer("color", GL_FLOAT, 3*sizeof(GLfloat), 3, sizeof(Vertex));
	shader.enableAttributeArray("color");
	shader.setAttributeBuffer("normal", GL_FLOAT, 7*sizeof(GLfloat), 3, sizeof(Vertex));
	shader.enableAttributeArray("normal");
	mesh.vertexBuff.release();

	// draw the mesh here
	glDrawElements(GL_TRIANGLES, mesh.numIndices, GL_UNSIGNED_SHORT, mesh.indices);

	// clean up
	shader.disableAttributeArray("vertex");
	shader.disableAttributeArray("color");
	shader.disableAttributeArray("normal");

	shader.release();
}

void GLWindow::renderSkeleton( Bone* root )
{
	if(!root) return; // empty skeleton
	QVector<MeshData> meshes = root->getMeshData();
	for (int i = 0; i < meshes.size(); ++i)
	{
		renderMesh(lightingShaderProgram, meshes[i], root->m_globalTransform);
	}
	
	for (int i = 0; i < root->childCount(); ++i)
	{
		renderSkeleton(root->getChild(i));
	}

}

void GLWindow::setupLights()
{

}





void GLWindow::keyPressEvent( QKeyEvent * e )
{
	switch (e->key())
	{
	case Qt::Key_F11:
		if ( isFullScreen() )
		{
			showNormal();
			resizeToScreenCenter();
		}
		else
		{
			showFullScreen();
		}
		updateGL();
		break;
	case Qt::Key_Escape:
		close();
	}
	
}

void GLWindow::resizeToScreenCenter()
{
	setGeometry( 0, 0, 1024, 768 );
	move(QApplication::desktop()->screen()->rect().center() - rect().center());
}

void GLWindow::updateLoop()
{
	lightAngle += 0.5;
	while (lightAngle >= 360) {
		lightAngle -= 360;
	}
	
	checkKeyState();
	updateGL();

}

void GLWindow::mousePressEvent( QMouseEvent *e )
{
	m_camera.LastMousePosition(vec2(e->pos()));
	e->accept();
}

void GLWindow::mouseMoveEvent( QMouseEvent *e)
{
	if (e->buttons() == Qt::RightButton)
	{
		m_camera.lookAround(vec2(e->x(), e->y()));
		updateGL();
	}
	
	m_camera.LastMousePosition(vec2(e->pos()));
	
	e->accept();
}

void GLWindow::wheelEvent( QWheelEvent *e )
{
	int delta = e->delta();

	if (e->orientation() == Qt::Vertical) {
		if (delta < 0) {
			m_camera.moveBackward();
		} else if (delta > 0) {
			m_camera.moveForward();
		}

		updateGL();
	}

	e->accept();
}

void GLWindow::checkKeyState()
{
 	if (GetAsyncKeyState('W')) m_camera.moveForward();
 	if (GetAsyncKeyState('S')) m_camera.moveBackward();
 	if (GetAsyncKeyState('A')) m_camera.moveLeft();
 	if (GetAsyncKeyState('D')) m_camera.moveRight();
 	if (GetAsyncKeyState('R')) m_camera.moveUp();
 	if (GetAsyncKeyState('F')) m_camera.moveDown();
}
