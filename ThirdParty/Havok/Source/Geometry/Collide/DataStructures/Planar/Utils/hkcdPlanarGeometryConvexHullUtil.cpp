/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/Geometry/hkcdPlanarGeometry.h>
#include <Geometry/Collide/Algorithms/Triangle/hkcdTriangleUtil.h>
#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometryConvexHullUtil.h>
#include <Geometry/Internal/Algorithms/Distance/hkcdDistancePointLine.h>

#include <Common/Base/Container/FreeListArray/hkFreeListArray.h>

//
//	Implementation

namespace hkcdPlanarGeomHullImpl
{
	//
	//	Types

	HK_DECLARE_HANDLE(VertexId, hkUint16, 0xFFFF);
	HK_DECLARE_HANDLE(RidgeId, hkUint16, 0xFFFF);
	HK_DECLARE_HANDLE(FacetId, hkUint16, 0xFFFF);
	typedef hkcdPlanarGeometryPrimitives::Plane		Plane;
	typedef hkcdPlanarGeometryPrimitives::Vertex	Vertex;

	//
	//	Generic edge. A ridge is defined as the intersection of 2 facets.

	template <int DIMENSION>
	struct Ridge
	{
		public:

			/// Constructor
			HK_FORCE_INLINE Ridge()
			{
				m_facets[0] = FacetId::invalid();
				m_facets[1] = FacetId::invalid();
			}

			/// Attaches a facet to this ridge.
			HK_FORCE_INLINE void attachFacet(FacetId facetId)
			{
				const int idx = m_facets[0].isValid() ? 1 : 0;
				HK_ASSERT(0x44f4139d, !m_facets[idx].isValid());
				m_facets[idx] = facetId;
			}

			/// Detaches a facet from this ridge.
			HK_FORCE_INLINE void detachFacet(FacetId facetId)
			{
				const int idx = (m_facets[0] == facetId) ? 0 : 1;
				HK_ASSERT(0x7bbae3e3, m_facets[idx] == facetId);
				m_facets[idx] = FacetId::invalid();
			}

			/// Replaces the given adjacent face id with another one.
			HK_FORCE_INLINE void replaceFacet(FacetId oldFacetId, FacetId newFacetId)
			{
				const int idx = (m_facets[0] == oldFacetId) ? 0 : 1;
				HK_ASSERT(0x696a2468, m_facets[idx] == oldFacetId);
				m_facets[idx] = newFacetId;
			}

			/// Returns true if this ridge is valid.
			HK_FORCE_INLINE bool isValid() const;

			/// Returns the adjacent facet id distinct from the given facetId.
			HK_FORCE_INLINE FacetId getFacetNeighbour(FacetId facetId) const
			{
				const int idx = (m_facets[0] == facetId) ? 0 : 1;
				HK_ASSERT(0x769e255f, m_facets[idx] == facetId);
				return m_facets[1 - idx];
			}

			/// Returns true if the ridge has any adjacent facets.
			HK_FORCE_INLINE bool hasAdjacentFacets() const							{	return m_facets[0].isValid() || m_facets[1].isValid();	}

			/// Returns true if the ridge is defined with the given vertex.
			HK_FORCE_INLINE bool hasVertex(VertexId vid) const;

			/// Sets this ridge's vertices to the common vertices shared between ridge0 and ridge1.
			HK_FORCE_INLINE void setIntersection(const Ridge& ridge0, const Ridge& ridge1);

			/// Appends the ridge's vertex ids to the given array and returns the new array size. It will stop when the array reaches a maxNumVertsOut points.
			HK_FORCE_INLINE int getVertexIds(VertexId* vertsOut, int numVertsOut) const
			{
				for (int k = DIMENSION - 2; (k >= 0) && (numVertsOut < DIMENSION); k--)
				{
					// See if we already have vertex k.
					int i = numVertsOut - 1;
					for (; i >= 0; i--)
					{
						if ( vertsOut[i] == m_vertices[k] )
						{
							break;
						}
					}

					if ( i < 0 )
					{
						// We don't, add it!
						vertsOut[numVertsOut++] = m_vertices[k];
					}
				}

				return numVertsOut;
			}

			/// Sets the ridge's vertices. The number of vertices must be DIMENSIONS - 1.
			HK_FORCE_INLINE void setVertexIds(const VertexId (&vertices)[DIMENSION - 1]);

			/// Returns true if this facet shares vertices with the given facet
			HK_FORCE_INLINE bool intersectsWith(const Ridge& otherRidge) const;

			HK_FORCE_INLINE static void setEmpty(Ridge& element, hkUint32 next);
			HK_FORCE_INLINE static hkUint32 getNext(const Ridge& element)			{	return (hkUint32)element.m_timeStamp;						}

		public:

			FacetId m_facets[2];					///< Adjacent facets (only 2 at most)
			VertexId m_vertices[DIMENSION - 1];		///< Ridge vertices in the forward orientation
			int m_timeStamp;						///< Time-stamp, used in traversal
	};

	//
	//	Sets the ridge's vertices. The number of vertices must be DIMENSIONS - 1.

	template <> HK_FORCE_INLINE void Ridge<2>::setVertexIds(const VertexId (&vertsIn)[1])	{	m_vertices[0] = vertsIn[0];								}
	template <> HK_FORCE_INLINE void Ridge<3>::setVertexIds(const VertexId (&vertsIn)[2])	{	m_vertices[0] = vertsIn[0];	m_vertices[1] = vertsIn[1];	}

	//
	//	Returns true if the ridge has the given vertex

	template <> HK_FORCE_INLINE bool Ridge<2>::hasVertex(VertexId vid) const	{	return m_vertices[0] == vid;	}
	template <> HK_FORCE_INLINE bool Ridge<3>::hasVertex(VertexId vid) const	{	return (m_vertices[0] == vid) || (m_vertices[1] == vid);	}

	//
	//	Returns true if this ridge is valid.

	template <> HK_FORCE_INLINE bool Ridge<2>::isValid() const						{	return m_vertices[0].isValid();	}
	template <> HK_FORCE_INLINE bool Ridge<3>::isValid() const						{	return m_vertices[0].isValid() && m_vertices[1].isValid();	}
	template <> HK_FORCE_INLINE void Ridge<2>::setEmpty(Ridge& r, hkUint32 next)	{	r.m_vertices[0] = VertexId::invalid();	r.m_timeStamp = next;	}
	template <> HK_FORCE_INLINE void Ridge<3>::setEmpty(Ridge& r, hkUint32 next)	{	r.m_vertices[0] = r.m_vertices[1] = VertexId::invalid(); r.m_timeStamp = next;	}

	//
	//	Returns true if this facet shares vertices with the given facet

	template <> HK_FORCE_INLINE bool Ridge<2>::intersectsWith(const Ridge& /*otherRidge*/) const	{	return true;	}	// No 2D ridges will intersect, return true to make the algorithms work!
	template <> HK_FORCE_INLINE bool Ridge<3>::intersectsWith(const Ridge<3>& otherRidge) const		{	return otherRidge.hasVertex(m_vertices[0]) || otherRidge.hasVertex(m_vertices[1]);	}

	//
	//	Sets this ridge's vertices to the common vertices shared between ridge0 and ridge1.

	template <> HK_FORCE_INLINE void Ridge<2>::setIntersection(const Ridge& ridge0, const Ridge& ridge1)	{	HK_ASSERT(0x52b826ec, !ridge1.hasVertex(ridge0.m_vertices[0]));	}	// We shouldn't be called with identical ridges!
	template <> HK_FORCE_INLINE void Ridge<3>::setIntersection(const Ridge& ridge0, const Ridge& ridge1)
	{
		const int idx = ridge1.hasVertex(ridge0.m_vertices[0]) ? 0 : 1;
		HK_ASSERT(0x7934fa59, ridge1.hasVertex(ridge0.m_vertices[idx]));
		m_vertices[0] = ridge0.m_vertices[idx];
	}

	//
	//	Generic face

	template <int DIMENSION>
	struct Facet
	{
		public:

			HK_FORCE_INLINE Facet()	: m_allocated(true)						{}
			HK_FORCE_INLINE hkBool32 isValid() const						{	return m_allocated;								}
			HK_FORCE_INLINE static void setEmpty(Facet& f, hkUint32 next)	{	f.m_allocated = false; f.m_timeStamp = next;	}
			HK_FORCE_INLINE static hkUint32 getNext(const Facet& f)			{	return (hkUint32)f.m_timeStamp;					}

		public:

			hkArray<VertexId> m_outsideVerts;			///< Array of vertex indices in the outside set.
			hkArray<RidgeId> m_ridges;					///< Array of face ridges, oriented CCW.
			Plane m_outwardPlane;						///< Facet normal pointing outwards (from an interior point).
			int m_timeStamp;							///< Time-stamp, used to track facet traversals
			hkBool m_allocated;
	};

	/// Mesh
	template <int DIMENSION>
	struct Mesh
	{
		public:

			typedef Ridge<DIMENSION> RidgeType;
			typedef Facet<DIMENSION> FacetType;

		public:

			/// Adds a vertex
			HK_FORCE_INLINE VertexId addVertex(hkIntVectorParameter iv)
			{
				const VertexId vid(m_vertices.getSize());
				m_vertices.expandOne().set(iv);
				return vid;
			}

			/// Adds a ridge having the given vertices.
			HK_FORCE_INLINE RidgeId addRidge(const VertexId (&vertices)[DIMENSION - 1])
			{
				const RidgeId ridgeId = m_ridges.allocate();
				m_ridges[ridgeId].setVertexIds(vertices);
				return ridgeId;
			}

			//
			//	Returns the vertices of the given face. It will only return the first DIMENSION vertices, as they are
			//	sufficient to determine the facet normal

			HK_FORCE_INLINE void getFacetVertexIds(FacetId fid, VertexId (&faceVertsOut)[DIMENSION]) const
			{
				int numVertsOut			= 0;
				const FacetType& facet	= m_facets[fid];
				const int numRidges		= facet.m_ridges.getSize();

				for (int ei = 0; (ei < numRidges) && (numVertsOut < DIMENSION); ei++)
				{
					const RidgeId ridgeId	= facet.m_ridges[ei];
					const RidgeType& ridge	= m_ridges[ridgeId];

					numVertsOut = ridge.getVertexIds(faceVertsOut, numVertsOut);
				}
			}

			/// Computes the outward normal of the given triangle
			HK_FORCE_INLINE void computeOutwardNormal(const VertexId (&vtxIdx)[DIMENSION], Plane& normalOut)
			{
				// Get vertices
				const hkIntVector vA	= m_vertices[vtxIdx[0].value()].getExactPosition();
				const hkIntVector vB	= m_vertices[vtxIdx[1].value()].getExactPosition();
				const hkIntVector vC	= m_vertices[vtxIdx[2].value()].getExactPosition();

				// Compute normal & offset
				hkIntVector vAB;		vAB.setSubS32(vB, vA);
				hkIntVector vAC;		vAC.setSubS32(vC, vA);
				hkInt64Vector4 iN;		iN.setCross(vAB, vAC);
				hkSimdInt<128> iO;		iO.setZero();	// NOT NECESSARY to compute the plane at this time!

				// Set plane equation
				normalOut.setExactEquation(iN, iO);
			}

			/// Adds a facet with the given ridges to the mesh.
			HK_FORCE_INLINE FacetId addFacet(const RidgeId (&ridges)[DIMENSION])
			{
				// Add the new facet
				FacetId facetId = m_facets.allocate();
				FacetType& facet = m_facets[facetId];

				// Add the facet to the ridges and the ridges to the facet
				facet.m_ridges.setSize(DIMENSION);
				for (int i = DIMENSION - 1; i >= 0; i--)
				{
					const RidgeId ridgeId = ridges[i];
					m_ridges[ridgeId].attachFacet(facetId);
					facet.m_ridges[i] = ridgeId;
				}

				// Get facet vertices
				VertexId facetVertexIds[DIMENSION];
				getFacetVertexIds(facetId, facetVertexIds);
			
				// Compute and set outward normal
				computeOutwardNormal(facetVertexIds, facet.m_outwardPlane);

				// Check whether the interior point is at a positive distance from the triangle.
				// If so, reverse the normal direction.
				if ( isVertexAboveFacet(m_interiorPtId, facetId) )
				{
					facet.m_outwardPlane.setOpposite(facet.m_outwardPlane);
				}

				// Return the id of the newly added triangle
				return facetId;
			}

			/// Checks whether the given vertex is among the given face's vertices
			HK_FORCE_INLINE bool isFacetVertex(VertexId vid, FacetId fid) const
			{
				const FacetType& f = m_facets[fid];
				for (int ei = f.m_ridges.getSize() - 1; ei >= 0; ei--)
				{
					const RidgeId eid = f.m_ridges[ei];
					if ( m_ridges[eid].hasVertex(vid) )
					{
						return true;
					}
				}

				return false;
			}

			/// Tests whether a point is above the given face in 3D
			HK_FORCE_INLINE int isVertexAboveFacet(VertexId vid, FacetId fid) const
			{
				// Get a vertex on the facet
				const FacetType& f	= m_facets[fid];
				const RidgeType& e	= m_ridges[f.m_ridges[0]];
				const Vertex& vX	= m_vertices[vid.value()];
				const Vertex& vO	= m_vertices[e.m_vertices[0].value()];

				// Compute approx eqn first
				{
					hkVector4d vOX;				vOX.setSub(vX.getApproxPosition(), vO.getApproxPosition());
					const hkSimdDouble64 dot	= vOX.dot<3>(f.m_outwardPlane.getApproxEquation());
					hkSimdDouble64 absDot;		absDot.setAbs(dot);

					if ( absDot.isGreater(hkSimdDouble64::fromFloat(1.0e-7f)) )
					{
						return dot.isGreaterZero();
					}
				}

				// If we got here, we failed to decide using floating point. Revert to fixed precision
				{
					hkIntVector vOX;			vOX.setSubS32(vX.getExactPosition(), vO.getExactPosition());
					hkInt64Vector4 iN;			f.m_outwardPlane.getExactNormal(iN);
					const hkSimdInt<128> dot	= iN.dot<3>(vOX);

					hkVector4fComparison lez;	lez.setOr(dot.lessZero(), dot.equalZero());
					return !lez.anyIsSet();
				}
			}

			///	Computes the outside sets. This will assign any unassigned vertices to the faces they are exterior to.
			void computeOutsideSets(const VertexId* HK_RESTRICT vertexIds, int numVerts, const FacetId* HK_RESTRICT facetIds, int numFacets)
			{
				// Iterate over all vertices.
				for (int i = 0; i < numVerts; i++)
				{
					const VertexId vid = vertexIds[i];

					// Test it against all faces. If we find a face visible from the vertex,
					// then we can stop.
					for (int k = 0; k < numFacets; k++)
					{
						const FacetId fid = facetIds[k];

						// Test for point in face
						const int bOutsideVertex	= isVertexAboveFacet(vid, fid);
						const bool bFaceVertex		= isFacetVertex(vid, fid);

						if ( bFaceVertex || bOutsideVertex )
						{
							// Point is above or belonging to a face, stop!
							if ( bOutsideVertex )
							{
								m_facets[fid].m_outsideVerts.pushBack(vid);
							}
							break;
						}
					}
				}
			}

			/// Checks if vtxI is further away on the given direction than vtxJ
			HK_FORCE_INLINE VertexId getFurthestVertex(const hkInt64Vector4& iN, const hkVector4d& dN, VertexId vidA, VertexId vidB) const
			{
				const Vertex& vA = m_vertices[vidA.value()];
				const Vertex& vB = m_vertices[vidB.value()];

				// Try floating-point first
				{
					hkVector4d vDiff;			vDiff.setSub(vA.getApproxPosition(), vB.getApproxPosition());
					const hkSimdDouble64 dot	= vDiff.dot<3>(dN);
					hkSimdDouble64 absDot;		absDot.setAbs(dot);
					if ( absDot.isGreater(hkSimdDouble64::fromFloat(1.0e-7f)) )
					{
						return dot.isGreaterZero() ? vidA : vidB;
					}
				}

				// At this point we need to switch to exact arithmetic
				{
					hkIntVector vDiff;				vDiff.setSubS32(vA.getExactPosition(), vB.getExactPosition());
					const hkSimdInt<128> dot		= iN.dot<3>(vDiff);
					return dot.lessZero().anyIsSet() ? vidB : vidA;
				}
			}

			/// Computes the furthest vertex in the given direction
			VertexId getSupportVertex(const Plane& planeDir, const hkArray<VertexId>& vertexSet) const
			{
				const int numVerts		= vertexSet.getSize();
				hkInt64Vector4 iN;		planeDir.getExactNormal(iN);
				const hkVector4d& dN	= planeDir.getApproxEquation();
				VertexId bestId			= vertexSet[0];

				for (int i = 1; i < numVerts; i++)
				{
					bestId = getFurthestVertex(iN, dN, bestId, vertexSet[i]);
				}

				return bestId;
			}

			/// Collects all neighbours of the given face that are visible from the given vertex.
			HK_FORCE_INLINE void collectVisibleNeighbors(FacetId fid, VertexId vid, hkArray<FacetId>& neighboursOut) const
			{
				// Clear neighbours array
				neighboursOut.setSize(0);

				// Get neighbours from the ridges
				const FacetType& f = m_facets[fid];
				const int numRidges = f.m_ridges.getSize();
				for (int ei = 0; ei < numRidges; ei++)
				{
					const RidgeType& e = m_ridges[f.m_ridges[ei]];

					FacetId neighbourId = e.getFacetNeighbour(fid);
					if ( (neighboursOut.indexOf(neighbourId) < 0) && isVertexAboveFacet(vid, neighbourId) )
					{
						neighboursOut.pushBack(neighbourId);
					}
				}
			}

			/// Merges facet srcFacetId into facet dstFacetId.
			HK_FORCE_INLINE void mergeFacets(FacetId dstFacetId, FacetId srcFacetId)
			{
				FacetType& faceA = m_facets[dstFacetId];
				FacetType& faceB = m_facets[srcFacetId];

				// Move all exterior vertices from B to A
				faceA.m_outsideVerts.append(faceB.m_outsideVerts);
				faceB.m_outsideVerts.setSize(0);

				// Locate all ridges shared between A and B and remove them. For the remainder of
				// edges of B, change the B's face Id to A's face Id.
				{
					for (int ei = faceA.m_ridges.getSize() - 1; ei >= 0; ei--)
					{
						const RidgeId eidA = faceA.m_ridges[ei];
						const RidgeType& e = m_ridges[eidA];
						if ( e.getFacetNeighbour(dstFacetId) == srcFacetId )
						{
							// This is a common edge and must be removed from A.
							faceA.m_ridges.removeAt(ei);
						}
					}

					for (int ei = faceB.m_ridges.getSize() - 1; ei >= 0; ei--)
					{
						const RidgeId eidB = faceB.m_ridges[ei];
						RidgeType& e = m_ridges[eidB];

						if ( e.getFacetNeighbour(srcFacetId) == dstFacetId )
						{
							// This is a common edge and must be removed from B.
							// The edge has already been removed from the edges of A, so we can delete it completely!
							faceB.m_ridges.removeAt(ei);
							e.detachFacet(srcFacetId);
							e.detachFacet(dstFacetId);
							m_ridges.release(eidB);
						}
						else
						{
							// This is an edge between faceB and its neighbour. Must replace B with A and append it to
							// the faces of A.
							e.detachFacet(srcFacetId);
							e.attachFacet(dstFacetId);
							faceA.m_ridges.pushBack(eidB);
							faceB.m_ridges.removeAt(ei);
						}
					}
				}

				// Can remove face B
				m_facets.release(srcFacetId);
			}

			/// Sort the face's ridges so that each ridge intersects with the next.
			HK_FORCE_INLINE void sortFacetRidges(FacetId fid)
			{
				FacetType& f = m_facets[fid];
				hkArray<RidgeId>& facetRidges = f.m_ridges;

				// Get number of ridges.
				const int numRidges = facetRidges.getSize();
				HK_ASSERT(0x6e3e3326, numRidges >= 2);

				// Start from the 2nd ridge. Try to connect each ridge with the precedent
				for (int ei = 1; ei < numRidges; ei++)
				{
					// Get previous ridge and its vertices
					RidgeId prevRidgeId = facetRidges[ei - 1];
					const RidgeType& prevRidge = m_ridges[prevRidgeId];

					// We must locate an edge that starts with vid1 and put it instead of the edge at ei
					int ej = ei;
					for (; ej < numRidges; ej++)
					{
						RidgeId currentRidgeId = facetRidges[ej];
						const RidgeType& currentRidge = m_ridges[currentRidgeId];

						if ( prevRidge.intersectsWith(currentRidge) )
						{
							break;	// Found a ridge that can be linked to prevRidge, stop!
						}
					}

					// See if we found something. If not, assert!
					HK_ASSERT(0x40acd662, ej < numRidges);

					// Swap edges ei and ej
					{
						const RidgeId tmpRidgeId = facetRidges[ei];
						facetRidges[ei] = facetRidges[ej];
						facetRidges[ej] = tmpRidgeId;
					}
				}

				// Final check. Last edge must connect with first
#ifdef HK_DEBUG
				{
					const RidgeType& firstRidge = m_ridges[facetRidges[0]];
					const RidgeType& lastRidge = m_ridges[facetRidges[numRidges - 1]];
					HK_ASSERT(0x491bbeaa, firstRidge.intersectsWith(lastRidge));
				}
#endif
			}

			/// Creates a new ridge from two existing ridges and a new vertex. The given ridges are assumed to be intersecting.
			HK_FORCE_INLINE RidgeId createSharedRidge(RidgeId eid0, RidgeId eid1, VertexId vid)
			{
				// Alloc the new ridge
				RidgeId newRidgeId = m_ridges.allocate();

				// Set-up its vertices
				RidgeType& newRidge = m_ridges[newRidgeId];
				const RidgeType& oldRidge0 = m_ridges[eid0];
				const RidgeType& oldRidge1 = m_ridges[eid1];
				newRidge.setIntersection(oldRidge0, oldRidge1);
				newRidge.m_vertices[DIMENSION - 2] = vid;

				// Return the ridge id
				return newRidgeId;
			}

			/// Creates a cone from the given vertex and the edges of the given face
			void createConeAroundFacet(FacetId boundaryFacetId, VertexId vid, hkArray<FacetId>& coneFacesOut)
			{
				const int numRidges = m_facets[boundaryFacetId].m_ridges.getSize();

				// Create last new ridge, between facets (n-1, 0).
				RidgeId lastNewRidgeId = createSharedRidge(m_facets[boundaryFacetId].m_ridges[numRidges - 1], m_facets[boundaryFacetId].m_ridges[0], vid);

				RidgeId facetRidges[3];
				facetRidges[0] = lastNewRidgeId;

				// Check special case for 2D
// 				if ( numRidges == 2 )
// 				{
// 					//			(DIMENSION == 2);
// 					//			G_ASSERT(numRidges == 2);
// 
// 					facetRidges[1]	= m_facets[boundaryFacetId].m_ridges[0];
// 					m_ridges[facetRidges[1]].detachFacet(boundaryFacetId);
// 					coneFacesOut.pushBack(addFacet(facetRidges, 2));
// 
// 					facetRidges[1]	= m_facets[boundaryFacetId].m_ridges[1];
// 					m_ridges[facetRidges[1]].detachFacet(boundaryFacetId);
// 					coneFacesOut.pushBack(addFacet(facetRidges, 2));
// 				}
//				else
				{
					HK_ASSERT(0x22744914, numRidges > 2);

					// Create each new facet
					for (int ei = 0; ei < numRidges - 1; ei++)
					{
						// Create new ridge between facets (ei, ei + 1).
						facetRidges[1]	= m_facets[boundaryFacetId].m_ridges[ei];
						facetRidges[2]	= createSharedRidge(facetRidges[1], m_facets[boundaryFacetId].m_ridges[ei + 1], vid);

						// Detach old face
						m_ridges[facetRidges[1]].detachFacet(boundaryFacetId);

						// Create new facet
						const FacetId triId = addFacet(facetRidges);
						coneFacesOut.pushBack(triId);
						facetRidges[0] = facetRidges[2];
					}

					// Create last facet
					{
						facetRidges[1] = m_facets[boundaryFacetId].m_ridges[numRidges - 1];
						facetRidges[2] = lastNewRidgeId;

						// Detach old face
						m_ridges[facetRidges[1]].detachFacet(boundaryFacetId);

						// Create new facet
						const FacetId triId = addFacet(facetRidges);
						coneFacesOut.pushBack(triId);
					}
				}

				// Remove all ridges from the current face
				m_facets[boundaryFacetId].m_ridges.setSize(0);
			}

			/// Removes a face
			HK_FORCE_INLINE void removeFacet(FacetId fid)
			{
				// Remove facer from all ridges
				FacetType& f = m_facets[fid];
				const int numEdges = f.m_ridges.getSize();
				for (int i = numEdges - 1; i >= 0; i--)
				{
					const RidgeId eid = f.m_ridges[i];
					RidgeType& e = m_ridges[eid];
					e.detachFacet(fid);

					if ( !e.hasAdjacentFacets() )
					{
						// We can delete the ridge entirely
						m_ridges.release(eid);
					}
				}

				// Remove facet
				m_facets.release(fid);
			}

		public:

			hkArray<Vertex> m_vertices;
			hkFreeListArray<RidgeType, RidgeId, -1, RidgeType> m_ridges;
			hkFreeListArray<FacetType, FacetId, -1, FacetType> m_facets;
			VertexId m_interiorPtId;
	};

	//
	//	Common specializations, for 2D and 3D.

	typedef Ridge<2>	Ridge2d;
	typedef Ridge<3>	Ridge3d;
	typedef Facet<2>	Facet2d;
	typedef Facet<3>	Facet3d;
	typedef Mesh<2>		Mesh2d;
	typedef Mesh<3>		Mesh3d;

	//
	//	Gets the furthest vertices in both negative and positive directions

	static void HK_CALL getSupportVertices(const hkVector4* HK_RESTRICT fVerts, int numVerts, hkVector4Parameter vDir, int& svMinOut, int& svMaxOut)
	{
		HK_ASSERT(0x120f2ee0, numVerts);

		// Init best indices
		hkIntVector idxP;	idxP.setZero();
		hkIntVector idxN	= idxP;
		hkIntVector idxCrt;	idxCrt.setAll(1);
		hkIntVector idxInc	= idxCrt;

		// Init best dots
		hkSimdReal vBestDotP = vDir.dot<3>(fVerts[0]);
		hkSimdReal vBestDotN = -vBestDotP;

		// Compute best dots
		for (int i = 1; i < numVerts; i++)
		{
			const hkSimdReal vDotP	= vDir.dot<3>(fVerts[i]);
			hkSimdReal vDotN		= -vDotP;

			idxP.setSelect(vDotP.greater(vBestDotP), idxCrt, idxP);
			idxN.setSelect(vDotN.greater(vBestDotN), idxCrt, idxN);
			vBestDotP.setMax(vDotP, vBestDotP);
			vBestDotN.setMax(vDotN, vBestDotN);
			idxCrt.setAddS32(idxCrt, idxInc);
		}

		// Output best indices
		svMinOut = idxN.getComponent<3>();
		svMaxOut = idxP.getComponent<3>();
	}

	//
	//	Computes the initial simplex. Returns the number of dimensions.

	static int HK_CALL computeInitialSimplex(const hkVector4* HK_RESTRICT fVerts, int numVerts, hkReal tolerance, int (&simplexVertsOut)[4])
	{
		int numSimplexDims = -1;
		if ( numVerts <= 0 )
		{
			return numSimplexDims;	// Error, no vertices!
		}

		// We have at least a vertex.
		simplexVertsOut[++numSimplexDims] = 0;

		// Check if we find a vertex far enough from the first vertex so we can create an edge
		const hkSimdReal tol	= hkSimdReal::fromFloat(tolerance);
		hkSimdReal tolSq;		tolSq.setMul(tol, tol);
		{
			const hkVector4 vA = fVerts[simplexVertsOut[0]];
			for (int i = 1; i < numVerts; i++)
			{
				hkVector4 vDiff;	vDiff.setSub(fVerts[i], vA);
				if ( tolSq.isLess(vDiff.lengthSquared<3>()) )
				{
					simplexVertsOut[++numSimplexDims] = i;
					break;
				}
			}
		}
		if ( numSimplexDims == 0 )
		{
			return numSimplexDims;	// Dimension is 0, we could not find 2 distinct vertices!
		}

		// We have at least 2 vertices, form a direction and choose the first two points
		// as the supporting vertices along that direction. Since there were no deletions so far, we can use
		// vertex ids 0 and 1 for this.
		{
			const hkVector4 vA	= fVerts[simplexVertsOut[0]];
			const hkVector4 vB	= fVerts[simplexVertsOut[1]];
			hkVector4 vAB;		vAB.setSub(vB, vA);

			getSupportVertices(fVerts, numVerts, vAB, simplexVertsOut[0], simplexVertsOut[1]);
		}

		// At this point we have an edge. Try to create an initial triangle, otherwise the data set has dimension 1!
		{
			// Compute the edge direction
			const hkVector4 vA		= fVerts[simplexVertsOut[0]];
			const hkVector4 vB		= fVerts[simplexVertsOut[1]];
			hkVector4 vX;			vX.setSub(vB, vA);
									vX.normalize<3>();

			// Build an orthogonal direction on vX
			hkMatrix3 m;			hkVector4Util::buildOrthonormal(vX, m);
			const hkVector4 vY		= m.getColumn<1>();

			// Get extremal vertices in direction Y
			int extremalVerts[2];
			getSupportVertices(fVerts, numVerts, vY, extremalVerts[0], extremalVerts[1]);

			// Compute distance from the points to the line
			hkSimdReal d0		= hkcdPointLineDistanceSquared(fVerts[extremalVerts[0]], vA, vB);
			hkSimdReal d1		= hkcdPointLineDistanceSquared(fVerts[extremalVerts[1]], vA, vB);
			hkSimdReal dMax;	dMax.setMax(d0, d1);

			if ( dMax.isLess(tolSq) )
			{
				// We found an axis along which the points are not distributed. Data has at most dimension 2
				// Check for vertices along axis Z.
				const hkVector4 vZ	= m.getColumn<2>();
				getSupportVertices(fVerts, numVerts, vZ, extremalVerts[0], extremalVerts[1]);
				d0 = hkcdPointLineDistanceSquared(fVerts[extremalVerts[0]], vA, vB);
				d1 = hkcdPointLineDistanceSquared(fVerts[extremalVerts[1]], vA, vB);

				dMax.setMax(d0, d1);
				if ( dMax.isLess(tolSq) )
				{
					// The points only have variance along the X axis, so 1D.
					return numSimplexDims;
				}

				// The points have only 2 dimensions.
				int newPointIdx = d1.isGreater(d0) ? 1 : 0;
				simplexVertsOut[++numSimplexDims] = extremalVerts[newPointIdx];
				return numSimplexDims;
			}

			// Data has at least dimension 2.
			int newPointIdx = d1.isGreater(d0) ? 1 : 0;
			simplexVertsOut[++numSimplexDims] = extremalVerts[newPointIdx];
		}

		// At this point we have a triangle. Try to create a tetrahedron, otherwise the data has dimension 2!
		{
			const hkVector4 vA	= fVerts[simplexVertsOut[0]];
			const hkVector4 vB	= fVerts[simplexVertsOut[1]];
			const hkVector4 vC	= fVerts[simplexVertsOut[2]];
			hkVector4 vN;		hkcdTriangleUtil::calcUnitNormal(vA, vB, vC, vN);

			int extremalVerts[2];
			getSupportVertices(fVerts, numVerts, vN, extremalVerts[0], extremalVerts[1]);
			hkVector4 vAP0;		vAP0.setSub(fVerts[extremalVerts[0]], vA);
			hkVector4 vAP1;		vAP1.setSub(fVerts[extremalVerts[1]], vA);
			hkSimdReal d0;		d0.setAbs(vN.dot<3>(vAP0));
			hkSimdReal d1;		d1.setAbs(vN.dot<3>(vAP1));
			hkSimdReal dMax;	dMax.setMax(d0, d1);

			if ( dMax.isLess(tol) )
			{
				// The points are confined into a plane, data dimension is 2!
				return numSimplexDims;
			}

			// Add new point, we are in 3D!
			int newPointIdx = d1.isGreater(d0) ? 1 : 0;
			simplexVertsOut[++numSimplexDims] = extremalVerts[newPointIdx];
		}

		return numSimplexDims;
	}

	//
	//	Builds the convex hull for the given mesh

	template <int DIMENSION>
	static void HK_CALL buildHull(const VertexId* HK_RESTRICT unassignedVertexIds, int numUnassignedVerts, const FacetId* HK_RESTRICT simplexFacetIds, Mesh<DIMENSION>& mesh)
	{
		typedef Facet<DIMENSION> FacetType;

		// Compute the outside sets.
		mesh.computeOutsideSets(unassignedVertexIds, numUnassignedVerts, simplexFacetIds, DIMENSION + 1);

		// Temporary array, used to collect the visible neighbors
		hkInplaceArray<FacetId, 256> visibleNeighbors;
		hkInplaceArray<FacetId, 256> coneFacets;

		// Iterate until we have no more outside vertices
		bool foundOutsideVerts;
		do 
		{
			foundOutsideVerts = false;

			// Iterate through all the facets.
			for (int fi = 0; fi < mesh.m_facets.getCapacity(); fi++)
			{
				// Get face, test for validity. If invalid or has no outside vertices, ignore!
				// Cannot cache the face, since it can change when adding new faces. We must use
				// only face ids.
				const FacetId fid		(fi);
				const FacetType& facet	= mesh.m_facets[fid];
				if ( !facet.isValid() || (facet.m_outsideVerts.getSize() < 1) )
				{
					continue;
				}

				// This face has outside vertices. Pick the furthest vertex.
				foundOutsideVerts = true;
				VertexId furthestVertexId = mesh.getSupportVertex(facet.m_outwardPlane, facet.m_outsideVerts);

				// While possible, collect neighbors of fid visible from furthestVertexId.
				do 
				{
					// Collect visible neighbours
					mesh.collectVisibleNeighbors(fid, furthestVertexId, visibleNeighbors);

					// Merge all into fid
					const int numNeighbours = visibleNeighbors.getSize();
					for (int ni = 0; ni < numNeighbours; ni++)
					{
						mesh.mergeFacets(fid, visibleNeighbors[ni]);
					}
				} while ( visibleNeighbors.getSize() );

				// Sort the face's edges so that each edge intersects with the next.
				mesh.sortFacetRidges(fid);

				// Create the cone
				coneFacets.setSize(0);
				mesh.createConeAroundFacet(fid, furthestVertexId, coneFacets);

				// Re-assign outVertices of face f.
				mesh.computeOutsideSets(mesh.m_facets[fid].m_outsideVerts.begin(), mesh.m_facets[fid].m_outsideVerts.getSize(), coneFacets.begin(), coneFacets.getSize());

				// Delete facet fid.
				mesh.removeFacet(fid);
			}
		} while ( foundOutsideVerts );
	}

	//
	//	Tests the hull
	template <int DIMENSION>
	static bool HK_CALL testHull(const Mesh<DIMENSION>& mesh)
	{
		const int numVerts = mesh.m_vertices.getSize();
		for (int i = 0; i < numVerts; i++)
		{
			const VertexId vid(i);

			// Locate a face the vertex is on
			for (int k = 0; k < mesh.m_facets.getCapacity(); k++)
			{
				const FacetId fid(k);
				if ( mesh.m_facets[fid].isValid() && mesh.isVertexAboveFacet(vid, fid) )
				{
					// We found a vertex that is not inside or on the hull!
					return false;
				}
			}
		}

		return true;
	}
}

//
//	Builds the convex hull from the given set of points

int HK_CALL hkcdPlanarGeometryConvexHullUtil::build(	const hkAabb& coordinateConversionAabb, const hkVector4* verticesIn, const int numVerticesIn, 
														hkRefPtr<hkcdPlanarGeometry>& convexHullOut, hkcdPlanarEntityDebugger* debugger)
{
	// Type shortcuts
	typedef hkcdPlanarGeomHullImpl::VertexId	VertexId;
	typedef hkcdPlanarGeomHullImpl::RidgeId		RidgeId;
	typedef hkcdPlanarGeomHullImpl::FacetId		FacetId;
	typedef hkcdPlanarGeomHullImpl::Mesh3d		Mesh3d;
	typedef hkcdPlanarGeomHullImpl::Facet3d		Facet3d;
	typedef hkcdPlanarGeomHullImpl::Ridge3d		Ridge3d;
	typedef hkcdPlanarGeometry::Plane			Plane;

	// Create the geometry
	hkcdPlanarGeometryPlanesCollection* convexHullPlanes = new hkcdPlanarGeometryPlanesCollection(coordinateConversionAabb);
	convexHullOut.setAndDontIncrementRefCount(new hkcdPlanarGeometry(convexHullPlanes, 0, debugger));
	convexHullPlanes->removeReference();

	// Determine initial simplex
	HK_ALIGN16 (int) simplexVerts[4];
	const hkReal vtxEps	= (hkReal)(2.0 / convexHullPlanes->getPositionScale().getReal());
	const int hullDims	= hkcdPlanarGeomHullImpl::computeInitialSimplex(verticesIn, numVerticesIn, vtxEps, simplexVerts);
	if ( hullDims < 0 )
	{
		return hullDims;	// Failed to determine a valid dimension!
	}

	// Special cases (0D, 1D). Either a single point or two points, already located by computeSimplex.
	switch ( hullDims )
	{
	case 3:	// 3D case.
		{
			// Set-up mesh from initial simplex
			Mesh3d mesh;

			// Add vertices
			for (int k = 0; k < numVerticesIn; k++)
			{
				hkIntVector iv;	convexHullPlanes->convertWorldPosition(verticesIn[k], iv);
				mesh.addVertex(iv);
			}

			// Compute an interior point inside the initial simplex.
			{
				hkIntVector iMid;					iMid.setAddS32(mesh.m_vertices[simplexVerts[0]].getExactPosition(), mesh.m_vertices[simplexVerts[1]].getExactPosition());
													iMid.setAddS32(iMid, mesh.m_vertices[simplexVerts[2]].getExactPosition());
													iMid.setAddS32(iMid, mesh.m_vertices[simplexVerts[3]].getExactPosition());
				const hkVector4Comparison iSigns	= iMid.lessZeroS32();
													iMid.setAbsS32(iMid);
													iMid.setShiftRight32<2>(iMid);
													iMid.setFlipSignS32(iMid, iSigns);
				mesh.m_interiorPtId = mesh.addVertex(iMid);
			}

			// Create ridges
			const VertexId ridgeAB[2] = { VertexId(simplexVerts[0]), VertexId(simplexVerts[1]) };		const RidgeId eAB = mesh.addRidge(ridgeAB);
			const VertexId ridgeBC[2] = { VertexId(simplexVerts[1]), VertexId(simplexVerts[2]) };		const RidgeId eBC = mesh.addRidge(ridgeBC);
			const VertexId ridgeCA[2] = { VertexId(simplexVerts[2]), VertexId(simplexVerts[0]) };		const RidgeId eCA = mesh.addRidge(ridgeCA);
			const VertexId ridgeAD[2] = { VertexId(simplexVerts[0]), VertexId(simplexVerts[3]) };		const RidgeId eAD = mesh.addRidge(ridgeAD);
			const VertexId ridgeBD[2] = { VertexId(simplexVerts[1]), VertexId(simplexVerts[3]) };		const RidgeId eBD = mesh.addRidge(ridgeBD);
			const VertexId ridgeCD[2] = { VertexId(simplexVerts[2]), VertexId(simplexVerts[3]) };		const RidgeId eCD = mesh.addRidge(ridgeCD);
			
			// Create facets
			const RidgeId facetABC[3] = { eAB, eBC, eCA };												const FacetId fABC = mesh.addFacet(facetABC);
			const RidgeId facetABD[3] = { eAB, eBD, eAD };												const FacetId fABD = mesh.addFacet(facetABD);
			const RidgeId facetBCD[3] = { eBC, eCD, eBD };												const FacetId fBCD = mesh.addFacet(facetBCD);
			const RidgeId facetCAD[3] = { eCA, eAD, eCD };												const FacetId fCAD = mesh.addFacet(facetCAD);
			
			// Build the hull
			hkLocalArray<VertexId> unassignedVertices(numVerticesIn);
			{
				unassignedVertices.setSize(numVerticesIn);
				for (int k = numVerticesIn - 1; k >= 0; k--)
				{
					unassignedVertices[k] = VertexId(k);
				}
				hkIntVector tmp;	tmp.load<4>((hkUint32*)&simplexVerts[0]);
									tmp.setSortS32<4, HK_SORT_ASCENDING>(tmp);
									tmp.store<4>((hkUint32*)&simplexVerts[0]);
				unassignedVertices.removeAt(tmp.getComponent<3>());
				unassignedVertices.removeAt(tmp.getComponent<2>());
				unassignedVertices.removeAt(tmp.getComponent<1>());
				unassignedVertices.removeAt(tmp.getComponent<0>());
			}

			const FacetId facets[4] = { fABC, fABD, fBCD, fCAD };
			hkcdPlanarGeomHullImpl::buildHull(unassignedVertices.begin(), unassignedVertices.getSize(), facets, mesh);
			HK_ASSERT(0x4221f72a, hkcdPlanarGeomHullImpl::testHull(mesh));
		
			// Create planar geometry from the hull
			hkcdPlanarGeometryPlanesCollection* hullPlanes = convexHullOut->accessPlanesCollection();

			for (int k = 0; k < mesh.m_facets.getCapacity(); k++)
			{
				const Facet3d& f = mesh.m_facets[FacetId(k)];
				if ( f.isValid() )
				{
					Plane p				= f.m_outwardPlane;
					const Ridge3d& r	= mesh.m_ridges[f.m_ridges[0]];
					hkInt64Vector4 iN;	p.getExactNormal(iN);
					hkSimdInt<128> iO;	iO.setNeg(iN.dot<3>(mesh.m_vertices[r.m_vertices[0].value()].getExactPosition()));
										p.setExactEquation(iN, iO);

					hullPlanes->addPlane(p);
				}
			}

			convexHullOut->weldPlanes();
		}
		break;

	default:	// Unhandled case
		return -1;
	}

	return hullDims;
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
