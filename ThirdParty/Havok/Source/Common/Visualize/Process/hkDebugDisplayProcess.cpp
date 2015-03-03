/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/Process/hkDebugDisplayProcess.h>
#include <Common/Visualize/Shape/hkDisplayCapsule.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeOStream.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Shape/hkDisplayGeometry.h>
#include <Common/Visualize/Shape/hkDisplaySphere.h>
#include <Common/Visualize/Shape/hkDisplayBox.h>
#include <Common/Visualize/Shape/hkDisplayAABB.h>
#include <Common/Visualize/Shape/hkDisplayCone.h>
#include <Common/Visualize/Shape/hkDisplayPlane.h>
#include <Common/Visualize/Shape/hkDisplaySemiCircle.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>

int hkDebugDisplayProcess::m_tag = 0;

hkProcess* HK_CALL hkDebugDisplayProcess::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkDebugDisplayProcess(); // doesn't need any context data
}

hkDebugDisplayProcess::hkDebugDisplayProcess()
	: hkProcess( true )
{
	if (m_tag == 0)
	{
		registerProcess();
	}
	hkDebugDisplay::getInstance().addDebugDisplayHandler(this);
}

void HK_CALL hkDebugDisplayProcess::registerProcess()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkDebugDisplayProcess::~hkDebugDisplayProcess()
{
	hkDebugDisplay::getInstance().removeDebugDisplayHandler(this);
}

hkResult hkDebugDisplayProcess::addGeometry(const hkArrayBase<hkDisplayGeometry*>& geometries, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint, hkGeometry::GeometryType geomType)
{
	return m_displayHandler->addGeometry(geometries, transform, id, tag, shapeIdHint);
}

hkResult hkDebugDisplayProcess::addGeometryInstance(hkUlong instID, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint)
{
	return m_displayHandler->addGeometryInstance(instID, transform, id, tag, shapeIdHint);
}

hkResult hkDebugDisplayProcess::setGeometryPickable( hkBool isPickable, hkUlong id, int tag )
{
	return m_displayHandler->setGeometryPickable(isPickable, id, tag);
}


hkResult hkDebugDisplayProcess::displayGeometry(const hkArrayBase<hkDisplayGeometry*>& geometries, const hkTransform& transform, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayGeometry(geometries, transform, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayGeometry(const hkArrayBase<hkDisplayGeometry*>& geometries, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayGeometry(geometries, color, id, tag);
}

hkResult hkDebugDisplayProcess::setGeometryVisibility(int geometryIndex, bool isVisible, hkUlong id, int tag)
{
	return m_displayHandler->setGeometryVisibility(geometryIndex, isVisible, id, tag);
}

hkResult hkDebugDisplayProcess::setGeometryColor(hkColor::Argb color, hkUlong id, int tag)
{
	return m_displayHandler->setGeometryColor(color, id, tag);
}

hkResult hkDebugDisplayProcess::setGeometryTransparency(float alpha, hkUlong id, int tag)
{
	return m_displayHandler->setGeometryTransparency(alpha, id, tag);
}

hkResult hkDebugDisplayProcess::updateGeometry(const hkTransform& transform, hkUlong id, int tag)
{
	return m_displayHandler->updateGeometry(transform, id, tag);
}

hkResult hkDebugDisplayProcess::updateGeometry( const hkMatrix4& transform, hkUlong id, int tag )
{
	return m_displayHandler->updateGeometry(transform, id, tag);
}

hkResult hkDebugDisplayProcess::skinGeometry(hkUlong* ids, int numIds, const hkMatrix4* poseModel, int numPoseModel, const hkMatrix4& worldFromModel, int tag )
{
	return m_displayHandler->skinGeometry(ids, numIds, poseModel, numPoseModel, worldFromModel, tag);
}

hkResult hkDebugDisplayProcess::removeGeometry(hkUlong id, int tag, hkUlong shapeIdHint)
{
	return m_displayHandler->removeGeometry(id, m_tag, shapeIdHint);
}

hkResult hkDebugDisplayProcess::updateCamera(const hkVector4& from, const hkVector4& to, const hkVector4& up, hkReal nearPlane, hkReal farPlane, hkReal fov, const char* name)
{
	return m_displayHandler->updateCamera(from, to, up, nearPlane, farPlane, fov, name);
}

hkResult hkDebugDisplayProcess::displayPoint(const hkVector4& position, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayPoint(position, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayLine(const hkVector4& start, const hkVector4& end, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayLine(start, end, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayTriangle(const hkVector4& a, const hkVector4& b, const hkVector4& c, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayTriangle(a,b,c, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayPoint2d(const hkVector4& position, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayPoint2d(position, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayLine2d(const hkVector4& start, const hkVector4& end, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayLine2d(start, end, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayTriangle2d(const hkVector4& a, const hkVector4& b, const hkVector4& c, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayTriangle2d(a,b,c, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayText2d(const char* text, const hkVector4& pos, hkReal sizeScale, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayText2d(text, pos, sizeScale, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayText(const char* text, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->displayText(text, color, id, tag);
}

hkResult hkDebugDisplayProcess::display3dText(const char* text, const hkVector4& pos, hkColor::Argb color, int id, int tag)
{
	return m_displayHandler->display3dText(text, pos, color, id, tag);
}

hkResult hkDebugDisplayProcess::displayAnnotation(const char* text, int id, int tag)
{
	return m_displayHandler->displayAnnotation(text, id, tag);
}

hkResult hkDebugDisplayProcess::sendMemStatsDump(const char* data, int length)
{
	return m_displayHandler->sendMemStatsDump(data, length);
}

hkResult hkDebugDisplayProcess::holdImmediate()
{
	return m_displayHandler->holdImmediate();
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
