/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKGP_INDEXED_MESH_H
#define HKGP_INDEXED_MESH_H

#include <Common/Internal/GeometryProcessing/Topology/hkgpTopology.h>
#include <Common/Internal/GeometryProcessing/AbstractMesh/hkgpAbstractMesh.h>
#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>

//
// Indexed mesh only store mesh topological informations, not geometrical ones.
//

/// Indexed mesh definitions
struct hkgpIndexedMeshDefinitions
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpIndexedMeshDefinitions);
	/// Vertex definitions.
	struct VertexBase {
		//+hk.MemoryTracker(ignore=True)
	};
	struct Vertex : public hkgpAbstractMeshDefinitions::Vertex<Vertex,VertexBase>
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,Vertex);
		HK_FORCE_INLINE int	compare(const Vertex& other) const { return m_index<other.m_index?-1:(m_index>other.m_index?+1:0); }
		hkUlong		m_index;		///< Source index or data.
		hkInt32		m_lidx;			///< Local index.
		hkInt32		m_numRefs;		///< Cardinality.
	};

	/// Triangle definitions.
	struct BaseTriangle {
		//+hk.MemoryTracker(ignore=True)
	};
	struct Triangle : public hkgpAbstractMeshDefinitions::Triangle<Triangle,BaseTriangle,Vertex*>
	{
		enum { INVALID_SET_ID = 0xffffffff };
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,Triangle);
		HK_FORCE_INLINE int	getNumRefs() const { return(vertex(0)->m_numRefs+vertex(1)->m_numRefs+vertex(2)->m_numRefs); }
		hkUlong		m_index;	///< Source index or data.
		hkInt32		m_set;		///< Set.
		hkInt32		m_material;	///< Source material.
		hkInt32		m_flags;	///< Flags.
		hkInt32		m_lidx;		///< Local index.
	};

	/// Edge definitions.
	struct Edge : public hkgpTopology::Edge<Edge,Vertex,Triangle>
	{
		Edge() : hkgpTopology::Edge<Edge,Vertex,Triangle>() {}
		Edge(Triangle* t,unsigned e) : hkgpTopology::Edge<Edge,Vertex,Triangle>(t,e) {}
		Edge(hkgpTopology::Uid id) : hkgpTopology::Edge<Edge,Vertex,Triangle>(id) {}
	};		
};

/// Indexed mesh implementation
class hkgpIndexedMesh : public hkgpAbstractMesh<hkgpIndexedMeshDefinitions::Edge, hkgpIndexedMeshDefinitions::Vertex, hkgpIndexedMeshDefinitions::Triangle>
{
public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY);
	//
	// Types
	//

	// Parent
	typedef hkgpAbstractMesh<hkgpIndexedMeshDefinitions::Edge, hkgpIndexedMeshDefinitions::Vertex, hkgpIndexedMeshDefinitions::Triangle>	Parent;

	// Flags
	struct Flags { enum _ {
		DEFAULT		=	0x00,
		BOUNDARY_0	=	0x01,
		BOUNDARY_1	=	0x02,
		BOUNDARY_2	=	0x04,
		DETACHED	=	0x08,
		MARKER		=	0x10,
	};};
	
	// Edge matching
	struct EdgeMatch
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, EdgeMatch);
		EdgeMatch() {}
		EdgeMatch(const Edge& e) : m_edge(e) {}
		HK_FORCE_INLINE hkUlong	getHash() const	{ return(hkGeometryProcessing::makeSymmetricHash(m_edge.start()->m_index,m_edge.end()->m_index)); }
		HK_FORCE_INLINE bool	operator==(const EdgeMatch& other) const { return(m_edge.canBind(other.m_edge)); }
		Edge		m_edge;
	};
	
	// By ref sorter.
	struct SortByAscendingReferences
	{
		//+hk.MemoryTracker(ignore=True)
		HK_FORCE_INLINE bool	operator()(const Triangle* a,const Triangle* b) const { return(a->getNumRefs()<b->getNumRefs()); }
	};

	// Strip generation configuration
	struct StripConfig
	{
		StripConfig() : m_minLength(4),m_maxLength((1<<16)-1),m_searchPasses(0) {}
		int	m_minLength;	///< Minimum allowed strip length
		int m_maxLength;	///< Maximum allowed strip length
		int m_searchPasses;	///< Best strip search passes
	};

	// Set infos
	struct SetInfos
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,SetInfos);
		hkArray<int>	m_links;		///< Direct connection to other sets.
	};

	// Edge barrier
	struct EdgeBarrier
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,EdgeBarrier);

		virtual ~EdgeBarrier() {}
		virtual bool isBlocked(Edge e) const { return false; }
	};

	// Vertex removal interface.
	struct IVertexRemoval
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,IVertexRemoval);

		virtual ~IVertexRemoval() {}

		virtual void	removeVertex(Vertex* vertex)=0;
	};

	// Triangle removal interface.
	struct ITriangleRemoval : public IVertexRemoval
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,ITriangleRemoval);

		virtual ~ITriangleRemoval() {}

		virtual void	removeTriangle(Triangle* triangle)=0;
	};

	// Edge collapse interface.
	struct IEdgeCollapse : public ITriangleRemoval
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,IEdgeCollapse);

		virtual ~IEdgeCollapse() {}

		virtual bool	validTriangleSet(const hkArrayBase<Triangle*>& triangles, const Vertex* from, const Vertex* to)=0;
		virtual void	bind(Edge e0, Edge e1)=0;
	};
	
	//
	// Methods
	//
	
	/// Constructor.
				hkgpIndexedMesh();

	/// Reset.
	void		reset();

	/// Append from another hkgpIndexedMesh.
	void		appendFromMesh(const hkgpIndexedMesh& mesh);

	/// Append hkGeometry to the mesh.
	void		appendFromGeometry(const hkGeometry& geometry);

	/// Append a single triangle to the mesh.
	Triangle*	appendTriangle(const hkGeometry::Triangle& triangle, hkUlong index);

	/// Append a single triangle to the mesh.
	Triangle*	appendTriangle(const hkUlong indices[3], hkUlong index, int material=0, int flags=Flags::DEFAULT);

	/// Collapse an edge.
	void		collapseEdge(Edge edge, Edge* edgeOut = HK_NULL, IEdgeCollapse* itrf = HK_NULL);

	/// Collapse an edge.
	void		collapseEdge(const hkArray<Triangle*>& ring, Edge edge, Edge* edgeOut = HK_NULL, IEdgeCollapse* itrf = HK_NULL);

	/// Check whether or not can an edge be flipped.
	hkBool		canFlipEdge(Edge edge) const;
	
	/// Flip an edge clockwise such as [edge.start() => edge.end()] become [edge.link().apex() => edge.apex()]
	Edge		flipEdge(Edge edge);

	/// Check whether or not can an edge be split.
	hkBool		canSplitEdge(Edge edge) const;

	/// Split an edge.
	Edge		splitEdge(Edge edge);

	/// Get 'start.ring' and 'end.ring'.
	void		getBothRings(Edge edge, hkArray<Triangle*>& ring) const;

	/// Get ring and returns cardinality, do not account for multiple rings sharing 'edge.start'.
	int			getReachableRing(Edge edge, hkArray<Triangle*>& ring) const;
	
	/// Remove a triangle.
	void		removeTriangle(Triangle* triangle, ITriangleRemoval* itrf = HK_NULL);
	
	/// Grow a strip.
	void		growStrip(Edge root,hkArray<Edge>& strip,const StripConfig& config) const;

	/// Generate strips.
	void		generateStrips(hkArray<hkArray<int> >& strips, hkArray<int>& leftOvers, hkArray<int>& map, const StripConfig& config=StripConfig()) const;

	/// Compute sets.
	void		computeSets(const EdgeBarrier& barriers = EdgeBarrier());

	/// Get all vertices belonging to a specified set.
	void		getSetVertices(int set, hkArray<Vertex*>& vertices) const;

	/// Get all triangles belonging to a specified set.
	void		getSetTriangles(int set, hkArray<Triangle*>& triangles) const;

	/// Delete all triangles belonging to a specified set.
	void		deleteSetTriangles(int set);

	//
	// Fields
	//

	hkPointerMap<hkUlong,Vertex*>				m_vMap;				///< index to vertex map.
	hkGeometryProcessing::HashTable<EdgeMatch>	m_eMap;				///< Open edges.
	hkArray<hkUlong>							m_invalidTriangles;	///< Triangle not added because invalid.
	int											m_nakedEdges;		///< Number of naked edges.
	hkArray<SetInfos>							m_sets;				///< Sets infos.
};

#include <Common/Internal/GeometryProcessing/IndexedMesh/hkgpIndexedMesh.inl>

#endif // HKGP_INDEXED_MESH_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
