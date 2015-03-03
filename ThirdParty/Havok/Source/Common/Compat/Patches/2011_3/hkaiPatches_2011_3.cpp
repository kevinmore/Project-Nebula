/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/KeyCode.h>

static void convertUp(hkDataObject& obj)
{
	const hkReal* vUp = obj["up"].asVec(4);

	hkUint16 pUp[3];
	for (int i=0; i<3; ++i)
	{
		hkReal biased = hkReal(30000) * hkReal(0x10000) * vUp[i];
		hkInt32 scaled = hkInt32(biased) + 0x80000000;
		pUp[i] = (scaled >> 16) & 0xffff;
	}
	hkDataArray packed = obj["new_up"].asArray();
	packed.setAll(pUp,3);
}

static void hkaiNavMeshPathSearchParameters_6_to_7(hkDataObject& obj)
{
	convertUp(obj);
}

static int _findStreamingSetIndex(hkDataArray& streamingSets, hkUint32 thisUid, hkUint32 oppositeUid)
{
	int numSets = streamingSets.getSize();
	for (int i=0; i<numSets; i++)
	{
		hkDataObject streamingSet = streamingSets[i].asObject();
		if( streamingSet["thisUid"].asInt() == (int)thisUid && streamingSet["oppositeUid"].asInt() == (int)oppositeUid)
		{
			return i;
		}
	}

	// Couldn't find set, so make a new one.
	streamingSets.setSize(numSets+1);
	hkDataObject newSet = streamingSets[numSets].asObject();
	newSet["thisUid"]     = (int) thisUid;
	newSet["oppositeUid"] = (int) oppositeUid;
	return numSets;
}

static void updateGraphAndVolume( hkDataObject& graphObj, bool isGraph )
{
	// The update for hkaiDirectedGraphExplicitCost is almost identical to that of hkaiNavVolume
	// The only differences are
	//	- Some different names of variabls (e.g. cell vs. node)
	//  - NavVolume::Edge doesn't have a cost member
	// So we'll define the strings accordingly, and use the same main code.
	// Since hkaiDirectedGraphExplicitCost patch function was written first, most of the local variables will use its terminology.
	const char* nodesStr        = isGraph ? "nodes"             : "cells";
	const char* nodeIndexStr    = isGraph ? "nodeIndex"         : "cellIndex";
	const char* edgeKeyStr      = isGraph ? "edgeKey"           : "edgeIndex";
	const char* targetUidStr    = isGraph ? "targetUid"         : "oppositeUid";
	const char* connectionsStr  = isGraph ? "graphConnections"  : "volumeConnections";
	const char* oppNodeIndexStr = isGraph ? "oppositeNodeIndex" : "oppositeCellIndex";
	const char* targetStr       = isGraph ? "target"            : "oppositeCell";
	const char* edgeStartStr    = isGraph ? "edgeStartIndex"    : "startEdgeIndex";

	const hkUint32 sectionUid = graphObj["sectionUid"].asInt();
	hkDataArray m_nodes = graphObj[nodesStr].asArray();
	hkDataArray m_edges = graphObj["edges"].asArray();
	hkDataArray m_externalEdges = graphObj["externalEdges"].asArray();

	hkDataArray m_streamingSets = graphObj["streamingSets"].asArray();

	// For directed graphs and nav volumes, we know which edges are external
	for (int i=0; i<m_externalEdges.getSize(); i++)
	{
		hkDataObject externalEdge = m_externalEdges[i].asObject();
		const int nodeIndex        = externalEdge[nodeIndexStr].asInt();
		const int edgeIndex        = externalEdge[edgeKeyStr  ].asInt();
		const hkUint32 oppositeUid = externalEdge[targetUidStr].asInt();

		hkDataObject edge = m_edges[edgeIndex].asObject();

		HK_ASSERT(0x5c718cb8, sectionUid != oppositeUid); // We know these are unequal, since it's in the externalEdge array
		{
			int idx = _findStreamingSetIndex(m_streamingSets, sectionUid, oppositeUid);
			hkDataObject streamingSet = m_streamingSets[idx].asObject();
			hkDataArray graphConnections = streamingSet[connectionsStr].asArray();

			// expandOne
			const int numConnections = graphConnections.getSize();
			graphConnections.setSize( numConnections + 1 );

			hkDataObject connection = graphConnections[numConnections].asObject();
			connection[nodeIndexStr   ] = nodeIndex;
			connection[oppNodeIndexStr] = edge[targetStr];

			// Mark the edge as invalid. We'll do another pass to remove these edges
			edge[targetStr] = -1;
		}
	}

	// Now clean up each node's set of edges. We'll have some "orphaned" edges, but that's not terrible.
	for (int n=0; n<m_nodes.getSize(); n++)
	{
		hkDataObject node = m_nodes[n].asObject();

		int startEdgeIndex = node[edgeStartStr].asInt();
		int numEdges = node["numEdges"].asInt();

		// Count backwards so we can remove safely
		//for (e = startEdgeIndex; e<startEdgeIndex + numEdges; e++)
		for (int e = startEdgeIndex + numEdges - 1; e >= startEdgeIndex; e--)
		{
			hkDataObject edge = m_edges[e].asObject();
			int edgeTarget = edge[targetStr].asInt();

			if (edgeTarget == -1)
			{
				// Copy the last edge here
				hkDataObject edgeSrc = m_edges[startEdgeIndex + numEdges - 1].asObject();
				edge[targetStr] = edgeSrc[targetStr];
				edge["flags" ] = edgeSrc["flags" ];
				if (isGraph)
				{
					edge["cost"  ] = edgeSrc["cost"  ];
				}

				// Reduce the number of edges
				numEdges--;
			}
		}

		node["numEdges"] = numEdges;
	}
}

static void hkaiDirectedGraphExplicitCost_5_to_6( hkDataObject& graphObj )
{
	updateGraphAndVolume(graphObj, true);
}

static void hkaiNavMeshInstance_2_to_3( hkDataObject& instance )
{
	hkDataObject originalMesh = instance["originalMesh"].asObject();
	if ( !originalMesh.isNull() )
	{
		instance["sectionUid"] = originalMesh["sectionUid"];
		instance["runtimeId" ] = originalMesh["runtimeId" ];
	}

}

static void hkaiNavMesh_12_to_13( hkDataObject& navMeshObj )
{
	const hkUint32 sectionUid = navMeshObj["sectionUid"].asInt();
	hkDataArray m_faces = navMeshObj["faces"].asArray();
	hkDataArray m_edges = navMeshObj["edges"].asArray();

	hkDataArray m_streamingSets = navMeshObj["streamingSets"].asArray();

	for (int f=0; f<m_faces.getSize(); f++)
	{
		hkDataObject face = m_faces[f].asObject();

		int startEdgeIndex = face["startEdgeIndex"].asInt();
		int numEdges = face["numEdges"].asInt();

		for (int e = startEdgeIndex; e < startEdgeIndex + numEdges; e++)
		{
			hkDataObject edge = m_edges[e].asObject();
			const hkUint32 oppositeUid = edge["oppositeSectionUid"].asInt();

			if ((oppositeUid != hkUint32(-1)) && (sectionUid != oppositeUid) )
			{
				int idx = _findStreamingSetIndex(m_streamingSets, sectionUid, oppositeUid);
				hkDataObject streamingSet = m_streamingSets[idx].asObject();
				hkDataArray navMeshConnections = streamingSet["meshConnections"].asArray();

				// expandOne
				const int numConnections = navMeshConnections.getSize();
				navMeshConnections.setSize( numConnections + 1 );

				hkDataObject connection = navMeshConnections[numConnections].asObject();
				connection["faceIndex"]         = f;
				connection["edgeIndex"]         = e;
				connection["oppositeFaceIndex"] = edge["oppositeFace"];
				connection["oppositeEdgeIndex"] = edge["oppositeEdge"];

				// Set the edge to be a boundary
				edge["oppositeFace"] = -1;
				edge["oppositeEdge"] = -1;
			}
		}
	}
}

static void hkaiNavVolume_9_to_10( hkDataObject& graphObj )
{
	updateGraphAndVolume(graphObj, false);
}

static void hkaiNavVolumePathSearchParameters_1_to_2(hkDataObject& obj)
{
	convertUp(obj);
}

static void hkaiCharacter_24_to_25(hkDataObject& characterObj)
{
	bool useNewAvoidance = characterObj["useNewAvoidance"].asInt() != 0;
	characterObj["avoidanceSolverType"] = (useNewAvoidance ? 0 : 1);
}

static void hkaiAvoidanceSolverMovementProperties_4_to_5(hkDataObject& obj)
{
	bool useAngularConstraints = obj["useAngularConstraints"].asInt() != 0;
	const int CONSTRAINTS_LINEAR_AND_ANGULAR = 1, CONSTRAINTS_LINEAR_ONLY = 2;
	obj["kinematicConstraintType"] = useAngularConstraints ? CONSTRAINTS_LINEAR_AND_ANGULAR : CONSTRAINTS_LINEAR_ONLY;

}

static void hkaiPlaneVolume_1_to_2(hkDataObject& obj)
{
	// equivalent to hkAabb.setEmpty();
	hkDataObject aabb = obj["aabb"].asObject();
	hkVector4 vMin; vMin.setAll( hkSimdReal_Max);
	hkVector4 vMax; vMax.setAll(-hkSimdReal_Max);
	aabb["min"] = vMin;
	aabb["max"] = vMax;
	
}

static void hkaiPathfindingUtilFindPathInput_9_to_10(hkDataObject& obj)
{
	hkDataObject params = obj["searchParameters"].asObject();
	params["searchSphereRadius" ] = obj["searchSphereRadius" ];
	params["searchCapsuleRadius"] = obj["searchCapsuleRadius"];
}


// Registration function is at the end of the file

void HK_CALL registerAiPatches_2011_3(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2011_3/hkaiPatches_2011_3.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
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
