#<make-shader "show tex on saq"
#:vertex-shader #{
    #version 430 core
    in vec3 in_pos;
	out vec2 tc;
    void main() {
        gl_Position = vec4(in_pos.xy, -0.5, 1);   
		tc = in_pos.xy * 0.5 + 0.5;
    }
}
#:fragment-shader #{
    #version 430 core
    uniform layout(binding=0) sampler2D tex;
	in vec2 tc;
    out vec4 out_col;
    void main() {
        out_col = vec4(texture(tex, vec2(1.0-tc.x,tc.y)).rgb,1);
    }
}
#:inputs (list "in_pos")
#:uniforms (list )>

