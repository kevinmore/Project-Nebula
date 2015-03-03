/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>
#include <Common/Base/System/Io/OStream/hkOStream.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>

#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>

#include <Physics2012/Utilities/Collide/ShapeUtils/SimpleMeshTklStreamer/hkpSimpleMeshTklStreamer.h>

hkpSimpleMeshShape* HK_CALL hkpTklStreamer::readStorageMeshFromTklStream(hkIstream &inputStream)
{
	hkpSimpleMeshShape* newShape = new hkpSimpleMeshShape();
	int numVertices = 0;
	inputStream >> numVertices;
	HK_ASSERT2(0x27f13f71,  numVertices > 2, "Less than three vertices, invalid tkl file" );
	newShape->m_vertices.setSize( numVertices );	

	HK_ALIGN_REAL(hkFloat32 xyz[4]);
	xyz[3] = 0.0f;
	for ( int v_it = 0; v_it < numVertices; v_it++ )
	{
		inputStream >> xyz[0];
		inputStream >> xyz[1];
		inputStream >> xyz[2];
		newShape->m_vertices[v_it].load<4>(&xyz[0]);
	}

	int numTriangles = 0;
	inputStream >> numTriangles;
	HK_ASSERT2(0x698c0975,  numTriangles > 0, "Less than 1 triangle, invalid tkl file" );
	newShape->m_triangles.setSize( numTriangles );

	for ( int tr_it = 0; tr_it < numTriangles; tr_it++ )
	{
		inputStream >> newShape->m_triangles[tr_it].m_a;
		inputStream >> newShape->m_triangles[tr_it].m_b;
		inputStream >> newShape->m_triangles[tr_it].m_c;
	}
	return newShape;
}

void HK_CALL hkpTklStreamer::writeStorageMeshShapeToTklStream(hkpSimpleMeshShape* shape, hkOstream &outputStream)
{
	const hkArray<hkVector4>& vertices = shape->m_vertices;
	const hkArray<hkpSimpleMeshShape::Triangle>& triangles = shape->m_triangles;

	outputStream << vertices.getSize() << "\n";
	HK_ALIGN_REAL(hkFloat32 xyz[4]);
	for ( int v_it = 0; v_it < vertices.getSize(); v_it++ )
	{
		vertices[v_it].store<4>(&xyz[0]);
		outputStream << xyz[0] << " ";
		outputStream << xyz[1] << " ";
		outputStream << xyz[2] << "\n";
	}
	
	outputStream << triangles.getSize() << "\n";
	for ( int tri_it = 0; tri_it < triangles.getSize(); tri_it++ )
	{
		outputStream << triangles[tri_it].m_a << " ";
		outputStream << triangles[tri_it].m_b << " ";
		outputStream << triangles[tri_it].m_c;
		outputStream << "\n";
	}
}

hkpSimpleMeshShape* HK_CALL hkpTklStreamer::readStorageMeshFromBtklArchive(hkIArchive &inputArchive)
{
	hkpSimpleMeshShape* newShape = new hkpSimpleMeshShape();
	int numVertices = inputArchive.read32();
	HK_ASSERT2(0x64cbe9f5,  numVertices > 2, "Less than three vertices, invalid tkl file" );
	newShape->m_vertices.setSize( numVertices );
	
	HK_ALIGN_REAL(hkFloat32 xyz[4]);
	xyz[3] = 0.0f;
	for ( int v_it = 0; v_it < numVertices; v_it++ )
	{
		xyz[0] = inputArchive.readFloat32();
		xyz[1] = inputArchive.readFloat32();
		xyz[2] = inputArchive.readFloat32();
		newShape->m_vertices[v_it].load<4>(&xyz[0]);
	}
	
	int numTriangles = inputArchive.read32();
	HK_ASSERT2(0x6e7127e5,  numTriangles > 0, "Less than 1 triangle, invalid tkl file" );
	newShape->m_triangles.setSize( numTriangles );
	
	for ( int tr_it = 0; tr_it < numTriangles; tr_it++ )
	{
		newShape->m_triangles[tr_it].m_a = inputArchive.read32();
		newShape->m_triangles[tr_it].m_b = inputArchive.read32();
		newShape->m_triangles[tr_it].m_c = inputArchive.read32();
	}
	return newShape;
}

void HK_CALL hkpTklStreamer::writeStorageMeshShapeToBtklArchive(hkpSimpleMeshShape* shape, hkOArchive &outputArchive)
{
	const hkArray<hkVector4>& vertices = shape->m_vertices;
	const hkArray<hkpSimpleMeshShape::Triangle>& triangles = shape->m_triangles;
	
	outputArchive.write32( vertices.getSize() );
	HK_ALIGN_REAL(hkFloat32 xyz[4]);
	for ( int v_it = 0; v_it < vertices.getSize(); v_it++ )
	{
		vertices[v_it].store<4>(&xyz[0]);
		outputArchive.writeFloat32( xyz[0] );
		outputArchive.writeFloat32( xyz[1] );
		outputArchive.writeFloat32( xyz[2] );
	}
	
	outputArchive.write32( triangles.getSize() );
	for ( int tr_it = 0; tr_it < triangles.getSize(); tr_it++ )
	{
		outputArchive.write32( triangles[tr_it].m_a );
		outputArchive.write32( triangles[tr_it].m_b );
		outputArchive.write32( triangles[tr_it].m_c );
	}
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
