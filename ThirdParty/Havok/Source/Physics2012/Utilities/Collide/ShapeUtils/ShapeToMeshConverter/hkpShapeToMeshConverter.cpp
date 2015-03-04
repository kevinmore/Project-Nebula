/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

// Needed for the class reflection
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeConverter/hkpShapeConverter.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>

#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>
#include <Common/GeometryUtilities/Mesh/Utils/NormalCalculator/hkNormalCalculator.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>

// this
#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeToMeshConverter/hkpShapeToMeshConverter.h>


/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                             hkpShapeToMeshConverter

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */void hkpShapeToMeshConverter::computeBoxTextureCoords( const hkVector4& pos, const hkVector4& norm, hkVector4& uvOut, hkReal textureMapScale)
{
	hkVector4 tangent,binorm;
	int maxExtent = norm.getIndexOfMaxAbsComponent<3>();

    switch (maxExtent)
    {
        case 0:
        {
			tangent = hkVector4::getConstant<HK_QUADREAL_0100>();
            binorm = hkVector4::getConstant<HK_QUADREAL_0010>();
            break;
        }
        case 1:
        {
			tangent = hkVector4::getConstant<HK_QUADREAL_1000>();
			binorm = hkVector4::getConstant<HK_QUADREAL_0010>();
            break;
        }
        case 2:
        {
			tangent = hkVector4::getConstant<HK_QUADREAL_1000>();
			binorm = hkVector4::getConstant<HK_QUADREAL_0100>();
            break;
        }
		default:
		{
			HK_ASSERT(0x6161eb45,0);
			tangent.setZero();
			binorm.setZero();
			break;
		}
    }

	const hkSimdReal texMapScale = hkSimdReal::fromFloat(textureMapScale) * hkSimdReal_Inv2;
    uvOut.set( pos.dot<3>(tangent) * texMapScale, pos.dot<3>(binorm) * texMapScale, hkSimdReal_0, hkSimdReal_1);
}

/* static */hkMeshShape* hkpShapeToMeshConverter::createMeshShape( hkMeshSystem* system, const hkArray<hkVector4>& vertices, const hkArray<hkVector4>& normals, const hkArray<hkUint16>& triangleIndices, const hkMatrix4& texCoordTransform, hkMeshMaterial* material, hkBool createTangents)
{
	hkVertexFormat format;
	// the desired format
	format.addElement(hkVertexFormat::USAGE_POSITION, hkVertexFormat::TYPE_FLOAT32, 3);
	format.addElement(hkVertexFormat::USAGE_NORMAL, hkVertexFormat::TYPE_FLOAT32, 3);
	if ( createTangents )
	{
		format.addElement(hkVertexFormat::USAGE_TANGENT, hkVertexFormat::TYPE_FLOAT32, 3);
		format.addElement(hkVertexFormat::USAGE_BINORMAL, hkVertexFormat::TYPE_FLOAT32, 3);
	}
	format.addElement(hkVertexFormat::USAGE_TEX_COORD, hkVertexFormat::TYPE_FLOAT32, 2);
	format.makeCanonicalOrder();

	const int numVertices = vertices.getSize();
	const int numTriangles = triangleIndices.getSize() / 3;

	// Lets create a suitably large vertex buffer
	hkMeshVertexBuffer* vertexBuffer = system->createVertexBuffer(format, numVertices);
	vertexBuffer->getVertexFormat(format); // update to what is actually supported

	hkMeshVertexBuffer::LockInput input;
	input.m_lockFlags = hkMeshVertexBuffer::ACCESS_WRITE | hkMeshVertexBuffer::ACCESS_WRITE_DISCARD;
	hkMeshVertexBuffer::LockedVertices lockedVertices;

	HK_ON_DEBUG( hkMeshVertexBuffer::LockResult res = )
		vertexBuffer->lock(input, lockedVertices);
	HK_ASSERT(0x3445345, res == hkMeshVertexBuffer::RESULT_SUCCESS);

	// Set the vertex positions and normals (handed in)
	{
		hkLocalArray<hkFloat32> posCoords(4*numVertices);
		posCoords.setSize(4*numVertices);
		for (int i=0; i<numVertices; ++i)
		{
			vertices[i].store<4, HK_IO_NATIVE_ALIGNED>(&posCoords[4*i]);
		}
		vertexBuffer->setElementVectorArray(lockedVertices, format.findElementIndex(hkVertexFormat::USAGE_POSITION, 0), posCoords.begin());
	}
	{
		hkLocalArray<hkFloat32> nrmCoords(4*numVertices);
		nrmCoords.setSize(4*numVertices);
		for (int i=0; i<numVertices; ++i)
		{
			normals[i].store<4, HK_IO_NATIVE_ALIGNED>(&nrmCoords[4*i]);
		}
		vertexBuffer->setElementVectorArray(lockedVertices, format.findElementIndex(hkVertexFormat::USAGE_NORMAL, 0), nrmCoords.begin());
	}

	if (format.findElementIndex(hkVertexFormat::USAGE_TEX_COORD, 0) != -1) 
	{
		hkLocalArray<hkFloat32> texCoords(4*numVertices);
		texCoords.setSize(4*numVertices);
		hkLocalArray<hkVector4> texCoordsVec(numVertices);
		texCoordsVec.setSize(numVertices);

		for (int i = 0; i < numVertices; i++)
		{
			hkVector4 uv;
			computeBoxTextureCoords( vertices[i], normals[i], uv, 1);
	
			// To be compatible with previous version z is 1 and w is 0
			hkVector4 tmp; tmp.setPermutation<hkVectorPermutation::XYWZ>(uv);
			texCoordTransform.multiplyVector(tmp, texCoordsVec[i]);
			texCoordsVec[i].store<4,HK_IO_NATIVE_ALIGNED>(&texCoords[4*i]);
		}

		// Set the tex coords
		vertexBuffer->setElementVectorArray(lockedVertices, format.findElementIndex(hkVertexFormat::USAGE_TEX_COORD, 0), texCoords.begin());
	
		if ( (format.findElementIndex(hkVertexFormat::USAGE_TANGENT, 0) != -1) &&
			 (format.findElementIndex(hkVertexFormat::USAGE_BINORMAL, 0) != -1) )
		{
			hkLocalArray<hkVector4> tangents(numVertices);
			hkLocalArray<hkVector4> binormals(numVertices);
			hkNormalCalculator::calculateTangentSpaces(vertices, normals, texCoordsVec, triangleIndices, tangents, binormals);

			// Set the tangent frame
			{
				hkLocalArray<hkFloat32> tanCoords(4*numVertices);
				tanCoords.setSize(4*numVertices);
				for (int i=0; i<numVertices; ++i)
				{
					tangents[i].store<4, HK_IO_NATIVE_ALIGNED>(&tanCoords[4*i]);
				}
				vertexBuffer->setElementVectorArray(lockedVertices, format.findElementIndex(hkVertexFormat::USAGE_TANGENT, 0), tanCoords.begin());
			}
			{
				hkLocalArray<hkFloat32> bitanCoords(4*numVertices);
				bitanCoords.setSize(4*numVertices);
				for (int i=0; i<numVertices; ++i)
				{
					binormals[i].store<4, HK_IO_NATIVE_ALIGNED>(&bitanCoords[4*i]);
				}
				vertexBuffer->setElementVectorArray(lockedVertices, format.findElementIndex(hkVertexFormat::USAGE_BINORMAL, 0), bitanCoords.begin());
			}
		}
	}
	
	vertexBuffer->unlock(lockedVertices);

	// Produce it
	hkMeshSectionCinfo section;
	section.m_indexType = hkMeshSection::INDEX_TYPE_UINT16;
	section.m_indices = triangleIndices.begin();
	section.m_numPrimitives = numTriangles;
	section.m_primitiveType = hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST;
	section.m_material = material;
	section.m_vertexBuffer = vertexBuffer;
	section.m_transformIndex = -1;
	section.m_vertexStartIndex = -1;

	hkMeshShape* shape = system->createShape(&section, 1);
	vertexBuffer->removeReference();

	return shape;
}

/* static */hkMeshShape* hkpShapeToMeshConverter::convertShapeToMesh( hkMeshSystem* system, const hkpShape* physicsShape, hkReal foldFactor, const hkMatrix4& texCoordTransform, hkMeshMaterial* material, hkReal cosSmoothingAngle, hkBool createTangents)
{
	hkGeometry* geometry = hkpShapeConverter::toSingleGeometry( physicsShape );

#if 0
	{
		HK_REPORT("shape converter produced");
		HK_REPORT(geometry->m_vertices.getSize()<<" vertices:");
		for (int i=0; i<geometry->m_vertices.getSize(); ++i)
		{
			const hkVector4& v = geometry->m_vertices[i];
			HK_REPORT("("<<v(0)<<","<<v(1)<<","<<v(2)<<","<<v(3)<<")");
		}
		HK_REPORT(geometry->m_triangles.getSize()<<" triangles:");
		for (int i=0; i<geometry->m_triangles.getSize(); ++i)
		{
			const hkGeometry::Triangle& t = geometry->m_triangles[i];
			HK_REPORT("["<<t.m_a<<","<<t.m_b<<","<<t.m_c<<"]");
		}
	}
#endif

	hkFindUniquePositionsUtil uniqueVerticesUtil;
	hkLocalArray<hkUint16> indices(geometry->m_triangles.getSize() * 3);

	{
		indices.setSize(geometry->m_triangles.getSize() * 3);
		{
			const hkGeometry::Triangle* cur = geometry->m_triangles.begin();
			const hkGeometry::Triangle* end = geometry->m_triangles.end();
			hkUint16* indicesOut = indices.begin();

			for (; cur != end; cur++, indicesOut += 3)
			{
				HK_ASSERT(0x8d7292b7, cur->m_a >= 0 && cur->m_a < 0xffff);
				HK_ASSERT(0x8d7292b6, cur->m_b >= 0 && cur->m_b < 0xffff);
				HK_ASSERT(0x8d7292b5, cur->m_c >= 0 && cur->m_c < 0xffff);

				indicesOut[0] = hkUint16(uniqueVerticesUtil.addPosition(geometry->m_vertices[cur->m_a]));
				indicesOut[1] = hkUint16(uniqueVerticesUtil.addPosition(geometry->m_vertices[cur->m_b]));
				indicesOut[2] = hkUint16(uniqueVerticesUtil.addPosition(geometry->m_vertices[cur->m_c]));
			}
		}
	}

	hkArray<hkVector4> vertices;
	hkArray<hkVector4> normals;
	hkArray<hkUint16> newIndices;
	hkArray<hkUint16> originalIndices;

	//const hkReal foldFactor = 1.0f - 1e-1f;
	hkNormalCalculator::calculateSmoothedGeometry(uniqueVerticesUtil.m_positions, indices.begin(), indices.getSize() / 3, foldFactor, vertices, normals, newIndices, originalIndices);

#if 0
	{
		HK_REPORT("calcSmoothedGeometry produced");
		HK_REPORT(vertices.getSize()<<" vertices:");
		for (int i=0; i<vertices.getSize(); ++i)
		{
			const hkVector4& v = vertices[i];
			HK_REPORT("("<<v(0)<<","<<v(1)<<","<<v(2)<<","<<v(3)<<")");
		}
		HK_REPORT(normals.getSize()<<" normals:");
		for (int i=0; i<normals.getSize(); ++i)
		{
			const hkVector4& v = normals[i];
			HK_REPORT("("<<v(0)<<","<<v(1)<<","<<v(2)<<")");
		}
		HK_REPORT(newIndices.getSize()<<" indices:");
		for (int i=0; i<newIndices.getSize(); i+=3)
		{
			hkUint16* idx = &(newIndices[i]);
			HK_REPORT("["<<idx[0]<<","<<idx[1]<<","<<idx[2]<<"]");
		}
	}
#endif

	hkMeshShape* meshShape = createMeshShape(system, vertices, normals, newIndices, texCoordTransform, material, createTangents);

	delete geometry;
	return meshShape;
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
