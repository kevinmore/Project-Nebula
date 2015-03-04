/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>

#include <Common/SceneData/Graph/hkxNode.h>
#include <Common/SceneData/Mesh/hkxMesh.h>
#include <Common/SceneData/Mesh/hkxMeshSection.h>
#include <Common/SceneData/Mesh/hkxVertexBuffer.h>
#include <Common/SceneData/Mesh/hkxIndexBuffer.h>
#include <Common/SceneData/Material/hkxMaterial.h>

#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>

#include <Common/SceneData/Mesh/Channels/hkxVertexFloatDataChannel.h>
#include <Common/SceneData/Mesh/Channels/hkxVertexIntDataChannel.h>
#include <Common/SceneData/Mesh/Channels/hkxVertexSelectionChannel.h>
#include <Common/SceneData/Mesh/Channels/hkxVertexVectorDataChannel.h>
#include <Common/SceneData/Mesh/hkxMeshSectionUtil.h>

static inline hkFloat32* _floatPtrByIndex( const char* data, int stride, int index )
{
	return (hkFloat32*)(data + stride*index);
}

struct _TriangleTangentInfo
{
	hkxIndexBuffer* m_indexBuffer;
	int m_firstIndex;
	bool m_leftHanded;
	hkVector4 m_sVector;
	hkVector4 m_tVector;

	typedef unsigned short ThreeIndices[3];
	void getVertexIndices (ThreeIndices& vindexOut) const
	{
		unsigned short offset = (unsigned short) m_indexBuffer->m_vertexBaseOffset;

		if (m_indexBuffer->m_indices16.getSize() > 0)
		{
			vindexOut[0] = offset + m_indexBuffer->m_indices16[m_firstIndex];
			vindexOut[1] = offset + m_indexBuffer->m_indices16[m_firstIndex+1];
			vindexOut[2] = offset + m_indexBuffer->m_indices16[m_firstIndex+2];
		}
		else
		{
			vindexOut[0] = offset + (unsigned short) m_firstIndex;
			vindexOut[1] = vindexOut[0]+1;
			vindexOut[2] = vindexOut[0]+2;
		}
	}

	void setVertexIndices (const ThreeIndices& vindex)
	{
		unsigned short offset = (unsigned short) m_indexBuffer->m_vertexBaseOffset;

		if (m_indexBuffer->m_indices16.getSize() > 0)
		{
			m_indexBuffer->m_indices16[m_firstIndex] = vindex[0] - offset;
			m_indexBuffer->m_indices16[m_firstIndex+1] = vindex[1] - offset;
			m_indexBuffer->m_indices16[m_firstIndex+2] = vindex[2] - offset;
		}
		else
		{
			m_firstIndex = vindex[0] - offset;
		}

	}
};


class _SplitVertexMap
{
public:

	_SplitVertexMap()
	{

	}

	void findMirroredVertices (const hkArray<_TriangleTangentInfo>& triangles)
	{
		// Brute force approach by now
		for (int t1=0; t1<triangles.getSize(); t1++)
		{
			const _TriangleTangentInfo& tinfo1 = triangles[t1];
			unsigned short t1verts[3];
			tinfo1.getVertexIndices(t1verts);		

			for (int t2=t1+1; t2<triangles.getSize(); t2++)
			{
				const _TriangleTangentInfo& tinfo2 = triangles[t2];
				unsigned short t2verts[3];
				tinfo2.getVertexIndices(t2verts);		

				if (tinfo1.m_leftHanded != tinfo2.m_leftHanded)
				{
					// Candidate triangles (different handness)
					// Look for shared vertices
					for (int i1=0; i1<3; i1++)
					{
						for (int i2=0; i2<3; i2++)
						{
							if (t1verts[i1]==t2verts[i2])
							{
								const unsigned short originalVertex = t1verts[i1];

								// Shared vertex
								const int arrayPos = _findSplitVertex(originalVertex);
								if (arrayPos==-1) // New
								{
									SplitVertex splitVertex;
									splitVertex.m_originalIndex = originalVertex;
									splitVertex.m_leftHanded = tinfo1.m_leftHanded;
									splitVertex.m_mirroredTriangles.pushBack(t2);
									m_splitVertices.pushBack(splitVertex);
								}
								else
								{
									SplitVertex& splitVertex = m_splitVertices[arrayPos];

									if (tinfo1.m_leftHanded != splitVertex.m_leftHanded)
									{
										if (splitVertex.m_mirroredTriangles.indexOf(t1)==-1)
										{
											splitVertex.m_mirroredTriangles.pushBack(t1);
										}
									}	
									else
									{
										if (splitVertex.m_mirroredTriangles.indexOf(t2)==-1)
										{
											splitVertex.m_mirroredTriangles.pushBack(t2);
										}
									}
								}
							}
						}
					}
				}
			}
		}

		if (m_splitVertices.getSize()>0)
		{
			HK_REPORT("Splitting "<<m_splitVertices.getSize()<<" shared vertex(s)");
		}
		else
		{
			HK_REPORT("No vertices splitted");
		}
	}

	struct SplitVertex
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_EXPORT, SplitVertex );
		unsigned short m_originalIndex;
		hkArray<int> m_mirroredTriangles;
		bool m_leftHanded;

		SplitVertex () {}

		// we need an explicit copy constructor since the array's one is protected
		SplitVertex (const SplitVertex& other) 
		{

			m_originalIndex = other.m_originalIndex;
			m_mirroredTriangles = other.m_mirroredTriangles;
			m_leftHanded = other.m_leftHanded;
		}
	};

	hkArray<SplitVertex> m_splitVertices;


private:

	int _findSplitVertex (unsigned short originalIndex)
	{

		for (int i=0; i<m_splitVertices.getSize(); i++)
		{
			if (m_splitVertices[i].m_originalIndex == originalIndex) 
			{
				return i;
			}
		}

		return -1;
	}

};

static void _calculateTangentInfo ( const hkxMeshSection* meshSection , hkArray<_TriangleTangentInfo>& triangleInfoOut)
{

	// Temp storage:
	const hkxVertexBuffer* vb = meshSection->m_vertexBuffer;
	const hkxVertexDescription& descIn = vb->getVertexDesc();

	const hkxVertexDescription::ElementDecl* posDeclIn = descIn.getElementDecl(hkxVertexDescription::HKX_DU_POSITION, 0);
	int posStrideIn = posDeclIn->m_byteStride;

	int numTextureChannels = descIn.getUsageCount( hkxVertexDescription::HKX_DU_TEXCOORD );
	int sourceTextureChannel = numTextureChannels - 1;
	if (numTextureChannels < 1)
		return;

	// see if any texture channel in the section has a hint for bump or normal map
	if (meshSection->m_material)
	{
		for (int texStage = 0; texStage < meshSection->m_material->m_stages.getSize(); ++texStage)
		{ 
			if ((meshSection->m_material->m_stages[texStage].m_usageHint == hkxMaterial::TEX_NORMAL) ||
				(meshSection->m_material->m_stages[texStage].m_usageHint == hkxMaterial::TEX_BUMP) )
			{
				sourceTextureChannel = hkMath::min2<int>( texStage, numTextureChannels-1 );
				break;
			}
		}
	} 

	const hkxVertexDescription::ElementDecl* uvDeclIn = descIn.getElementDecl(hkxVertexDescription::HKX_DU_TEXCOORD, sourceTextureChannel);	
	if (!uvDeclIn)
	{
		return;
	}
	int uvStrideIn = uvDeclIn->m_byteStride;

	const char* posBufIn = static_cast<const char*>( vb->getVertexDataPtr( *posDeclIn ) );
	const char* uvBufIn  = static_cast<const char*>( vb->getVertexDataPtr( *uvDeclIn ) );

	hkSimdReal FAIL_TOLERANCE_SQRD; FAIL_TOLERANCE_SQRD.setFromFloat(hkReal(1e-10f));

	// For all triangles we know about:
	for (int pi=0; pi < meshSection->m_indexBuffers.getSize(); ++pi)
	{
		const hkxIndexBuffer* ib = meshSection->m_indexBuffers[pi];
		if (ib->m_indices16.getSize() == 0)
			continue; // not handled yet (XX todo)

		const unsigned short* indices = ib->m_indices16.begin();
		const unsigned short vbOffset = (unsigned short)( ib->m_vertexBaseOffset );
		const int numIndices = ib->m_length;

		int curIndex = 0;			
		unsigned short curTri[3] = {0,0,0};

		if (ib->m_indexType == hkxIndexBuffer::INDEX_TYPE_TRI_STRIP)
		{
			// prime the strip
			curTri[1] = indices? indices[0] : (vbOffset);
			curTri[2] = indices? indices[1] : (vbOffset + 1);
			curIndex = 2;
		}

		while (curIndex < numIndices)
		{
			int firstIndex;
			if (ib->m_indexType == hkxIndexBuffer::INDEX_TYPE_TRI_LIST)
			{
				firstIndex = curIndex;
				curTri[0] = (unsigned short)( indices? indices[curIndex++] : (vbOffset + curIndex++) );
				curTri[1] = (unsigned short)( indices? indices[curIndex++] : (vbOffset + curIndex++) );
				curTri[2] = (unsigned short)( indices? indices[curIndex++] : (vbOffset + curIndex++) );
			}
			else // tri strip. TODO: Can optimise this perhaps as some verts already computed.
			{
				firstIndex = curIndex - 2;
				curTri[0] = curTri[1]; curTri[1] = curTri[2];
				curTri[2] = (unsigned short)( indices? indices[curIndex++] : (vbOffset + curIndex++) );
			}

			if ((curTri[0] == curTri[1]) ||
				(curTri[0] == curTri[2]) ||
				(curTri[1] == curTri[2]) )
				continue; // degenerate

			hkVector4 v21,v31;
			{
				const hkFloat32* vv1 = _floatPtrByIndex( posBufIn, posStrideIn, curTri[0]);
				const hkFloat32* vv2 = _floatPtrByIndex( posBufIn, posStrideIn, curTri[1]);
				const hkFloat32* vv3 = _floatPtrByIndex( posBufIn, posStrideIn, curTri[2]);
				hkVector4 v1,v2,v3; 
				v1.load<3,HK_IO_NATIVE_ALIGNED>(vv1);
				v2.load<3,HK_IO_NATIVE_ALIGNED>(vv2);
				v3.load<3,HK_IO_NATIVE_ALIGNED>(vv3);
				v21.setSub(v2,v1);
				v31.setSub(v3,v1);
			}

			hkSimdReal s1,t1,s2,t2;
			{
				const hkFloat32* uv1 = _floatPtrByIndex( uvBufIn, uvStrideIn, curTri[0] );
				const hkFloat32* uv2 = _floatPtrByIndex( uvBufIn, uvStrideIn, curTri[1] );
				const hkFloat32* uv3 = _floatPtrByIndex( uvBufIn, uvStrideIn, curTri[2] );
				hkVector4 w1,w2,w3;
				w1.load<2,HK_IO_NATIVE_ALIGNED>(uv1);
				w2.load<2,HK_IO_NATIVE_ALIGNED>(uv2);
				w3.load<2,HK_IO_NATIVE_ALIGNED>(uv3);
				hkVector4 st1,st2;
				st1.setSub(w2,w1);
				st2.setSub(w3,w1);
				s1 = st1.getComponent<0>(); //w2.x - w1.x;
				t1 = st1.getComponent<1>(); //w2.y - w1.y;
				s2 = st2.getComponent<0>(); //w3.x - w1.x;
				t2 = st2.getComponent<1>(); //w3.y - w1.y;
			}

			hkSimdReal scale = (s1 * t2 - s2 * t1);
			hkSimdReal absScale; absScale.setAbs(scale);
			hkSimdReal r; r.setSelect(absScale.greaterZero(), scale.reciprocal(), hkSimdReal_1);

			hkVector4 sdir; 
			{
				hkVector4 v21t2; v21t2.setMul(v21,t2);
				hkVector4 v31t1; v31t1.setMul(v31,t1);
				sdir.setSub(v21t2, v31t1);
				sdir.mul(r);
				sdir.zeroComponent<3>();
			}

			hkVector4 tdir; 
			{
				hkVector4 v21s2; v21s2.setMul(v21,s2);
				hkVector4 v31s1; v31s1.setMul(v31,s1);
				tdir.setSub(v31s1, v21s2);
				tdir.mul(r);
				tdir.zeroComponent<3>();
			}

			if ( sdir.lengthSquared<3>().isLess(FAIL_TOLERANCE_SQRD) | tdir.lengthSquared<3>().isLess(FAIL_TOLERANCE_SQRD) )
			{
				// Constant U or V - ignore this triangle
				// Todo : count and report how many triangles ignored
				continue;
			}

			// normal
			hkVector4 normal;
			{
				normal.setCross(v21, v31);
			}

			hkVector4 sCrossT; sCrossT.setCross(sdir,tdir);

			_TriangleTangentInfo tinfo;
			tinfo.m_sVector = sdir;
			tinfo.m_tVector = tdir;
			hkSimdReal cross = sCrossT.dot<3>(normal);
			tinfo.m_leftHanded = bool(cross.isLessZero());
			tinfo.m_indexBuffer = meshSection->m_indexBuffers[pi];
			tinfo.m_firstIndex = firstIndex;
			triangleInfoOut.pushBack(tinfo);
		}
	}
}

static void _writeTangentData  ( const hkArray<_TriangleTangentInfo>& tangentInfo, hkxMeshSection* meshSection)
{
	hkxVertexBuffer* vb = meshSection->m_vertexBuffer;
	int numVerts = vb->getNumVertices();
	hkArray<hkVector4> tan1(numVerts, hkVector4::getZero());
	hkArray<hkVector4> tan2(numVerts, hkVector4::getZero());

	const hkxVertexDescription& desc = vb->getVertexDesc();
	const hkxVertexDescription::ElementDecl* normDecl = desc.getElementDecl( hkxVertexDescription::HKX_DU_NORMAL, 0);
	const hkxVertexDescription::ElementDecl* tangentDecl = desc.getElementDecl( hkxVertexDescription::HKX_DU_TANGENT, 0);
	const hkxVertexDescription::ElementDecl* binormalDecl = desc.getElementDecl( hkxVertexDescription::HKX_DU_BINORMAL, 0);

	const char* normIn = static_cast<const char*>( vb->getVertexDataPtr(*normDecl) );
	char* tangentOut = static_cast<char*>( vb->getVertexDataPtr(*tangentDecl) );
	char* binormalOut = static_cast<char*>( vb->getVertexDataPtr(*binormalDecl) );
	int normStrideIn = normDecl->m_byteStride;
	int tangentStrideOut = tangentDecl->m_byteStride;
	int binormalStrideOut = binormalDecl->m_byteStride;

	// Add the tangents for each vertex
	{
		for (int tri=0; tri<tangentInfo.getSize(); tri++)
		{
			const _TriangleTangentInfo& tinfo = tangentInfo[tri];

			_TriangleTangentInfo::ThreeIndices triIndices;
			tinfo.getVertexIndices(triIndices);

			for (int tvert=0; tvert<3; tvert++)
			{
				tan1[ triIndices[tvert] ].add( tinfo.m_sVector );
				tan2[ triIndices[tvert] ].add( tinfo.m_tVector );
			}
		}
	}

	hkSimdReal FAIL_TOLERANCE_SQRD; FAIL_TOLERANCE_SQRD.setFromFloat(hkReal(1e-10f));

	hkVector4 nDnt;
	hkVector4 nCt;

	int numPartialFail = 0;
	int numCompleteFail = 0;

	for (int a = 0; a < numVerts; a++)
	{
		const hkFloat32* np = _floatPtrByIndex(normIn, normStrideIn, a);
		hkVector4 n; n.load<3,HK_IO_NATIVE_ALIGNED>(np);
		const hkVector4& t1 = tan1[a];
		const hkVector4& t2 = tan2[a];

		hkSimdReal nt1 = t1.lengthSquared<3>();
		hkSimdReal nt2 = t2.lengthSquared<3>();

		hkBool32 useT1 = nt1.isGreater( FAIL_TOLERANCE_SQRD );
		if (!useT1)
		{
			++numPartialFail;	
		}

		const hkVector4& t = ( useT1? t1 : t2 );
		const hkVector4& bt = ( useT1? t2 : t1 );

		hkFloat32* tangentp = _floatPtrByIndex( useT1? tangentOut : binormalOut, useT1? tangentStrideOut : binormalStrideOut, a);
		hkFloat32* bitangentp = _floatPtrByIndex( useT1? binormalOut : tangentOut, useT1? binormalStrideOut : tangentStrideOut, a);
		hkVector4 tangent; tangent.load<3,HK_IO_NATIVE_ALIGNED>(tangentp);
		hkVector4 bitangent; bitangent.load<3,HK_IO_NATIVE_ALIGNED>(bitangentp);

		// Gram-Schmidt orthogonalize
		// tangent[a] = (t - n * (n * t)).Normalize();
		hkSimdReal nDt = n.dot<3>(t);
		nDnt.setMul(nDt, n);
		tangent.setSub(t, nDnt);
		hkSimdReal tlen = tangent.lengthSquared<3>();
		if ( tlen.isLess(FAIL_TOLERANCE_SQRD) )
		{
			// EXP-525
			// The UV coordinates are constant so we can't create binormal/tangents based on them
			// Pick arbitrary orthogonal vectors
			++numCompleteFail;
			if (!useT1)
			{
				--numPartialFail;	// don't report it twice
			}	

			hkVector4Util::calculatePerpendicularVector(n, tangent);
			bitangent.setCross(tangent,n);
		}
		else
		{
			tangent.mul( tlen.sqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>() ); 
		}

		// Calculate handedness
		// tangent[a].w = (n % t * tan2[a] < 0.0F) ? -1.0F : 1.0F;
		nCt.setCross( n, t );
		const hkSimdReal tw = nCt.dot<3>(bt);

		// Calculate the bitangent
		if ( tw.isLessZero() )
		{
			bitangent.setCross( tangent, n );
		}
		else
		{
			bitangent.setCross( n, tangent );
		}

		// write back
		tangent.zeroComponent<3>();
		tangent.store<4,HK_IO_NATIVE_ALIGNED>(tangentp);
		bitangent.zeroComponent<3>();
		bitangent.store<4,HK_IO_NATIVE_ALIGNED>(bitangentp);

	} // forall verts

	if (numPartialFail > 0)
	{
		HK_WARN_ALWAYS(0xabba98bc, "Found " << numPartialFail << " degenerate Tangents, caused by very small UV mapping along U direction. ");
	}
	if (numCompleteFail > 0)
	{
		HK_WARN_ALWAYS(0xabba98bd, "Found " << numCompleteFail << " degenerate basis, caused by bad UV mapping, had to pick arbitary tangents. ");
	}
}

struct IndexPair
{
	int fromLocal; //anim vb with anim dif in it
	int toOrig; //new mesh vb index for resultant vb to map to
};

static void HK_CALL _fixupVertexAnimations(hkxMeshSection* meshSection, hkArray<_SplitVertexMap::SplitVertex>& splitVertices )
{
	int numVertsBeforeSplit = meshSection->m_vertexBuffer->getNumVertices() - splitVertices.getSize();

	hkArray<_SplitVertexMap::SplitVertex*> oldIndexToSplit( numVertsBeforeSplit, HK_NULL );
	hkArray<int> oldIndexToNewVertIndex( numVertsBeforeSplit, 0 );
	for (int si = 0; si< splitVertices.getSize(); ++si )
	{
		oldIndexToSplit[ splitVertices[si].m_originalIndex ] = &splitVertices[si];
		oldIndexToNewVertIndex[ splitVertices[si].m_originalIndex ] = si + numVertsBeforeSplit;
	}

	for (int vi =0; vi < meshSection->m_vertexAnimations.getSize(); ++vi)
	{   
		hkxVertexAnimation* anim = meshSection->m_vertexAnimations[vi];

		hkArray<IndexPair> needCopy;

		// if anim affects old index, then it should affect new vertex the same 
		for (int ai=0; ai < anim->m_vertexIndexMap.getSize(); ++ai)
		{
			int origIndex = anim->m_vertexIndexMap[ai];
			_SplitVertexMap::SplitVertex* sp = oldIndexToSplit[origIndex];
			if (sp)
			{
				IndexPair& ip = needCopy.expandOne();
				ip.fromLocal = ai;
				ip.toOrig = oldIndexToNewVertIndex[ origIndex ];
			}
		}

		// expand out the anim by needCopy.size
		if (needCopy.getSize() > 0)
		{
			//xx would be nice if the vb had a non destructive resize..
			hkxVertexAnimation* newanim = new hkxVertexAnimation();
			newanim->m_time = anim->m_time;
			newanim->m_componentMap.append( anim->m_componentMap );
			int newNumAnimVerts = anim->m_vertData.getNumVertices() + needCopy.getSize();
			newanim->m_vertexIndexMap.reserve( newNumAnimVerts );
			newanim->m_vertexIndexMap.append( anim->m_vertexIndexMap );
			newanim->m_vertData.setNumVertices(newNumAnimVerts, anim->m_vertData.getVertexDesc());
			newanim->m_vertData.copy( anim->m_vertData, false );
			for (int ci=0; ci < needCopy.getSize(); ++ci)
			{
				const IndexPair& it = needCopy[ci];
				int newLocalIndex = anim->m_vertData.getNumVertices() + ci;
				newanim->m_vertData.copyVertex( anim->m_vertData, it.fromLocal, newLocalIndex);
				newanim->m_vertexIndexMap.pushBack(it.toOrig);
			}
			meshSection->m_vertexAnimations[vi].setAndDontIncrementRefCount(newanim);
		}
	}
}

void HK_CALL hkxMeshSectionUtil::computeTangents( hkxMesh* mesh, bool splitVertices, const char* nameHint )
{
	const bool createChannel = splitVertices;

	if (createChannel)
	{
		// Add a vertex selection channel with the split vertices
		hkxMesh::UserChannelInfo* newchannel = new hkxMesh::UserChannelInfo;
		newchannel->m_name = "Mirrored UV Split Vertices";
		newchannel->m_className = hkxVertexSelectionChannelClass.getName();

		mesh->m_userChannelInfos.pushBack(newchannel);
		newchannel->removeReference();
	}

	for (int s =0; s < mesh->m_sections.getSize(); ++s)
	{
		hkxMeshSection* section = mesh->m_sections[s];
		hkxVertexBuffer* vb = section->m_vertexBuffer;

		hkxVertexSelectionChannel* splitVertschannel = HK_NULL;
		if (createChannel)
		{
			splitVertschannel = new hkxVertexSelectionChannel;
			hkRefVariant var (splitVertschannel);
			section->m_userChannels.pushBack(var);
			splitVertschannel->removeReference();
		}

		hkStringBuf title;
		if (nameHint)
		{
			title.printf("Processing mesh \"%s\", section %d", nameHint, s);
		}
		else
		{
			title.printf("Processing unnamed mesh, section %d", s);
		}
		HK_REPORT_SECTION_BEGIN(0xf1de98a2, title.cString());

		const hkxVertexDescription& inDesc = vb->getVertexDesc();
		// If this section already has tangents then skip it
		if ( (inDesc.getElementDecl( hkxVertexDescription::HKX_DU_BINORMAL, 0) != HK_NULL) ||
			(inDesc.getElementDecl( hkxVertexDescription::HKX_DU_TANGENT, 0) != HK_NULL) )
		{
			HK_REPORT("*Tangent information already present, ignoring*");
			HK_REPORT_SECTION_END();
			continue;
		}

		// If the current format has no texture coords we can't create tangents
		if ( inDesc.getElementDecl( hkxVertexDescription::HKX_DU_TEXCOORD, 0) == HK_NULL ) 
		{
			HK_WARN_ALWAYS(0xabba98bb, "No UV texture coordinates found, can't generate tangents");
			HK_REPORT_SECTION_END();
			continue;
		}

		// Start by looking at tangent information for the current vertex buffer
		hkArray<_TriangleTangentInfo> tangentInfo;
		_calculateTangentInfo(section, tangentInfo);

		// Then look out for seams that need to be split (EXP-582)
		_SplitVertexMap splitVertexMap;

		if (splitVertices)
		{
			splitVertexMap.findMirroredVertices (tangentInfo);
		}

		const int numExtraVertices = splitVertexMap.m_splitVertices.getSize();

		// The format may already allow tangents. If so we allocate a new format
		// of the same type and fill in the tangent data, since we don't want to
		// affect unlisted meshes which share the same format
		hkxVertexDescription desiredDesc;
		for (int e=0; e < inDesc.m_decls.getSize(); ++e)
		{
			desiredDesc.m_decls.pushBack( inDesc.m_decls[e] );
		}

		if ( desiredDesc.getUsageCount( hkxVertexDescription::HKX_DU_TANGENT ) < 1 )
		{
			desiredDesc.m_decls.pushBack(hkxVertexDescription::ElementDecl(hkxVertexDescription::HKX_DU_TANGENT, hkxVertexDescription::HKX_DT_FLOAT, 3 ) );
		}

		if ( desiredDesc.getUsageCount( hkxVertexDescription::HKX_DU_BINORMAL ) < 1)
		{
			desiredDesc.m_decls.pushBack(hkxVertexDescription::ElementDecl(hkxVertexDescription::HKX_DU_BINORMAL, hkxVertexDescription::HKX_DT_FLOAT, 3 ) );
		}

		// Always copy
		{
			const int originalNumberOfVertices = vb->getNumVertices();
			const int newNumberOfVertices = originalNumberOfVertices + numExtraVertices;

			hkxVertexBuffer* newVb = new hkxVertexBuffer();
			newVb->setNumVertices( newNumberOfVertices, desiredDesc );
			newVb->copy( *vb, false );

			for (int vi=0; vi<splitVertexMap.m_splitVertices.getSize(); vi++)
			{
				const _SplitVertexMap::SplitVertex& splitVertex = splitVertexMap.m_splitVertices[vi];
				const unsigned short oldIndex = splitVertex.m_originalIndex;
				const unsigned short newIndex = (unsigned short) (originalNumberOfVertices +  vi);
				newVb->copyVertex( *vb, oldIndex, newIndex);

				if (splitVertschannel)
				{
					splitVertschannel->m_selectedVertices.pushBack(oldIndex);
					// new index will automatically be added as part of the general
					// channel update
				}
				// Now replace the indices in the mirrored triangles
				for (int ti=0; ti<splitVertex.m_mirroredTriangles.getSize(); ti++)
				{
					_TriangleTangentInfo& mirroredTri = tangentInfo[splitVertex.m_mirroredTriangles[ti]];
					_TriangleTangentInfo::ThreeIndices originalIndices;
					mirroredTri.getVertexIndices(originalIndices);

					_TriangleTangentInfo::ThreeIndices newIndices;
					for (int j=0; j<3; j++)
					{
						if (originalIndices[j]==oldIndex)
						{
							newIndices[j] = newIndex;
						}
						else
						{
							newIndices[j] = originalIndices[j];
						}
					}

					mirroredTri.setVertexIndices(newIndices);
				}
			}

			if (splitVertices) // EXP-1828
			{
				// Go through vertex channels and update them accordingly
				for (int c=0; c<mesh->m_userChannelInfos.getSize(); ++c)
				{
					if (hkString::strCmp(mesh->m_userChannelInfos[c]->m_className, hkxVertexSelectionChannelClass.getName())==0)
					{
						hkxVertexSelectionChannel* channel = static_cast<hkxVertexSelectionChannel*> (section->m_userChannels[c].val());

						for (int vi=0; vi<splitVertexMap.m_splitVertices.getSize(); vi++)
						{
							const _SplitVertexMap::SplitVertex& splitVertex = splitVertexMap.m_splitVertices[vi];
							const hkUint32 oldIndex = splitVertex.m_originalIndex;
							const hkUint32 newIndex = originalNumberOfVertices +  vi;

							if (channel->m_selectedVertices.indexOf(oldIndex)>=0)
							{
								channel->m_selectedVertices.pushBack(newIndex);
							}
						}						
						// Sort the selected vertices in increasing order
						hkSort(channel->m_selectedVertices.begin(), channel->m_selectedVertices.getSize());
					}
					else if (hkString::strCmp(mesh->m_userChannelInfos[c]->m_className, hkxVertexFloatDataChannelClass.getName())==0)
					{
						hkxVertexFloatDataChannel* channel = static_cast<hkxVertexFloatDataChannel*> (section->m_userChannels[c].val());

						channel->m_perVertexFloats.setSize(newNumberOfVertices);

						for (int vi=0; vi<splitVertexMap.m_splitVertices.getSize(); vi++)
						{
							const _SplitVertexMap::SplitVertex& splitVertex = splitVertexMap.m_splitVertices[vi];
							const hkUint32 oldIndex = splitVertex.m_originalIndex;
							const hkUint32 newIndex = originalNumberOfVertices +  vi;
							channel->m_perVertexFloats[newIndex] = channel->m_perVertexFloats[oldIndex];
						}
					}
					else if (hkString::strCmp(mesh->m_userChannelInfos[c]->m_className, hkxVertexIntDataChannelClass.getName())==0)
					{
						hkxVertexIntDataChannel* channel = static_cast<hkxVertexIntDataChannel*> (section->m_userChannels[c].val());

						channel->m_perVertexInts.setSize(newNumberOfVertices);

						for (int vi=0; vi<splitVertexMap.m_splitVertices.getSize(); vi++)
						{
							const _SplitVertexMap::SplitVertex& splitVertex = splitVertexMap.m_splitVertices[vi];
							const hkUint32 oldIndex = splitVertex.m_originalIndex;
							const hkUint32 newIndex = originalNumberOfVertices +  vi;
							channel->m_perVertexInts[newIndex] = channel->m_perVertexInts[oldIndex];
						}
					}
					else if (hkString::strCmp(mesh->m_userChannelInfos[c]->m_className, hkxVertexVectorDataChannelClass.getName())==0)
					{
						hkxVertexVectorDataChannel* channel = static_cast<hkxVertexVectorDataChannel*> (section->m_userChannels[c].val());

						channel->m_perVertexVectors.setSize(4*newNumberOfVertices);

						for (int vi=0; vi<splitVertexMap.m_splitVertices.getSize(); vi++)
						{
							const _SplitVertexMap::SplitVertex& splitVertex = splitVertexMap.m_splitVertices[vi];
							const hkUint32 oldIndex = splitVertex.m_originalIndex;
							const hkUint32 newIndex = originalNumberOfVertices +  vi;
							channel->m_perVertexVectors[4*newIndex  ] = channel->m_perVertexVectors[4*oldIndex  ];
							channel->m_perVertexVectors[4*newIndex+1] = channel->m_perVertexVectors[4*oldIndex+1];
							channel->m_perVertexVectors[4*newIndex+2] = channel->m_perVertexVectors[4*oldIndex+2];
							channel->m_perVertexVectors[4*newIndex+3] = channel->m_perVertexVectors[4*oldIndex+3];
						}
					}
				}
			}

			section->m_vertexBuffer = newVb;
			newVb->removeReference();
		}

		_writeTangentData(tangentInfo, section);

		if (numExtraVertices > 0)
		{
			_fixupVertexAnimations( section, splitVertexMap.m_splitVertices );
		}

		HK_REPORT("Tangent information generated");

		HK_REPORT_SECTION_END();
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
